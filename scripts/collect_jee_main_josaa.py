
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = (
    "https://josaa.admissions.nic.in/"
    "applicant/seatmatrix/openingclosingrankarchieve.aspx"
)

OUTPUT_DIR = Path("data/raw/jee_main")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = {
    "year": "ctl00$ContentPlaceHolder1$ddlYear",
    "round": "ctl00$ContentPlaceHolder1$ddlroundno",
    "type": "ctl00$ContentPlaceHolder1$ddlInstype",
    "institute": "ctl00$ContentPlaceHolder1$ddlInstitute",
    "branch": "ctl00$ContentPlaceHolder1$ddlBranch",
    "seat": "ctl00$ContentPlaceHolder1$ddlSeatType",
    "submit": "ctl00$ContentPlaceHolder1$btnSubmit",
}

TYPE_VALUES = {
    "NIT": "NIT",
    "IIIT": "3IT",
    "GFTI": "CFI",
}

TABLE_ID = "ctl00_ContentPlaceHolder1_GridView1"

COLUMNS = [
    "exam",
    "year",
    "round",
    "institute_type",
    "institute",
    "academic_program",
    "quota",
    "seat_type",
    "gender",
    "opening_rank",
    "closing_rank",
    "source",
]


def make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Referer": URL,
    })
    # TLS certificate verification stays enabled.
    return session


def get_soup(session):
    response = session.get(URL, timeout=60)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def hidden_form_values(soup):
    data = {}

    for element in soup.select("input[type=hidden][name]"):
        data[element["name"]] = element.get("value", "")

    return data


def selected_dropdown_values(soup):
    data = {}

    for select in soup.select("select[name]"):
        selected = select.find("option", selected=True)

        if selected is None:
            selected = select.find("option")

        if selected is not None:
            data[select["name"]] = selected.get("value", "")

    return data


def postback(session, soup, field_name, value):
    """Select a dropdown option and process its ASP.NET postback."""
    data = hidden_form_values(soup)
    data.update(selected_dropdown_values(soup))
    data[field_name] = str(value)
    data["__EVENTTARGET"] = field_name
    data["__EVENTARGUMENT"] = ""

    response = session.post(URL, data=data, timeout=60)
    response.raise_for_status()
    time.sleep(0.15)

    return BeautifulSoup(response.text, "html.parser")


def get_options(soup, field_name):
    select = soup.find("select", attrs={"name": field_name})

    if select is None:
        return []

    result = []

    for option in select.find_all("option"):
        value = option.get("value", "").strip()
        label = option.get_text(" ", strip=True)

        if value and label and value not in ("0", "-1"):
            if label.lower() not in (
                "--select--",
                "select",
                "select an option",
            ):
                result.append((value, label))

    return result


def choose_option(options, wanted):
    """Find an option by exact value, or case-insensitive label."""
    wanted = str(wanted).strip().lower()

    for value, label in options:
        if value.strip().lower() == wanted:
            return value

        if label.strip().lower() == wanted:
            return value

    raise ValueError(
        f"Option {wanted!r} not found. "
        f"Available options: {options[:12]}"
    )


def setup_selection(session, year, round_no, type_code):
    """Load the page and select year, round, and institution type."""
    soup = get_soup(session)

    for key, wanted in (
        ("year", str(year)),
        ("round", str(round_no)),
        ("type", type_code),
    ):
        field = FIELDS[key]
        options = get_options(soup, field)
        value = choose_option(options, wanted)
        soup = postback(session, soup, field, value)

    return soup


def parse_results(soup, year, round_no, type_name):
    table = soup.find("table", id=TABLE_ID)

    if table is None:
        # Do not accidentally parse an unrelated page table.
        raise RuntimeError(
            f"Expected result table {TABLE_ID!r} was not found."
        )

    records = []

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        values = [c.get_text(" ", strip=True) for c in cells]

        # Result rows have seven cells; the header is skipped.
        if len(values) != 7:
            continue

        institute, program, quota, seat, gender, opening, closing = values

        if (
            not institute
            or not program
            or not opening
            or not closing
        ):
            continue

        # Keep rank strings intact, including a possible P suffix.
        records.append({
            "exam": "JEE Main",
            "year": int(year),
            "round": int(round_no),
            "institute_type": type_name,
            "institute": institute,
            "academic_program": program,
            "quota": quota,
            "seat_type": seat,
            "gender": gender,
            "opening_rank": opening,
            "closing_rank": closing,
            "source": URL,
        })

    return records


def is_cse_program(label):
    """Match Computer Science and Engineering programs only."""
    text = re.sub(r"\s+", " ", label).strip().lower()

    return (
        "computer science and engineering" in text
        and not text.startswith("--")
    )


def collect_institute(
    session,
    year,
    round_no,
    type_name,
    type_code,
    institute_value,
    institute_label,
):
    """Collect every matching CSE program at one institution."""
    soup = setup_selection(session, year, round_no, type_code)

    # Select the institute to load its available academic programs.
    soup = postback(
        session,
        soup,
        FIELDS["institute"],
        institute_value,
    )

    branches = [
        (value, label)
        for value, label in get_options(soup, FIELDS["branch"])
        if is_cse_program(label)
    ]

    if not branches:
        return [], "No matching CSE program"

    institute_rows = []

    for branch_value, branch_label in branches:
        try:
            # Rebuild the dropdown state before selecting each program.
            soup = setup_selection(
                session, year, round_no, type_code
            )
            soup = postback(
                session,
                soup,
                FIELDS["institute"],
                institute_value,
            )
            soup = postback(
                session,
                soup,
                FIELDS["branch"],
                branch_value,
            )

            # Select ALL categories, then submit the search form.
            soup = postback(
                session,
                soup,
                FIELDS["seat"],
                choose_option(
                    get_options(soup, FIELDS["seat"]),
                    "ALL",
                ),
            )

            data = hidden_form_values(soup)
            data.update(selected_dropdown_values(soup))
            data[FIELDS["submit"]] = "Submit"

            response = session.post(URL, data=data, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            
            rows = parse_results(soup, year, round_no, type_name)

            print(f"    DEBUG: selected branch = {branch_label!r}")
            print(f"    DEBUG: raw result rows = {len(rows)}")

            if rows:
                print(f"    DEBUG: first result program = "
                    f"{rows[0]['academic_program']!r}")
                print(f"    DEBUG: first result institute = "
                    f"{rows[0]['institute']!r}")

                rows = [
                    row for row in rows
                    if row["academic_program"].strip() == branch_label.strip()
                ]

            print(f"    DEBUG: rows after branch filter = {len(rows)}")


            institute_rows.extend(rows)
            time.sleep(0.2)

        except Exception as exc:
            print(
                f"  Program skipped: {institute_label} | "
                f"{branch_label} | {exc}"
            )

    return institute_rows, None


def main():
    print("JEE Main JoSAA CSE collector — all institutions")
    print("Initial scope: one year, one round, CSE programs only.")
    print("Source: official JoSAA archive")
    print()

    year = input("Year [2025]: ").strip() or "2025"
    round_no = input("Round [2]: ").strip() or "2"

    if not year.isdigit() or not round_no.isdigit():
        raise ValueError("Year and round must be numeric.")

    year = int(year)
    round_no = int(round_no)

    type_names = ["NIT", "IIIT", "GFTI"]
    all_rows = []

    for type_name in type_names:
        type_code = TYPE_VALUES[type_name]
        print(f"\n{'=' * 60}")
        print(f"Discovering {type_name} institutions...")
        print("=" * 60)

        try:
            soup = setup_selection(
                make_session(), year, round_no, type_code
            )
        except Exception as exc:
            print(f"Could not load {type_name} dropdown: {exc}")
            continue

        institutes = get_options(soup, FIELDS["institute"])
        institutes = [
            (value, label)
            for value, label in institutes
            if label.strip().upper() != "ALL"
        ]
        print(f"Institutions listed: {len(institutes)}")

        # Use one session per type to preserve the server cookies.
        session = make_session()

        try:
            # Establish a fresh session and verify the selected year/type.
            setup_selection(session, year, round_no, type_code)
        except Exception as exc:
            print(f"Could not initialise session: {exc}")
            continue

        type_rows = []
        failures = []

        for index, (institute_value, institute_label) in enumerate(
            institutes, start=1
        ):
            print(
                f"[{index}/{len(institutes)}] "
                f"{type_name}: {institute_label}"
            )

            try:
                rows, warning = collect_institute(
                    session=session,
                    year=year,
                    round_no=round_no,
                    type_name=type_name,
                    type_code=type_code,
                    institute_value=institute_value,
                    institute_label=institute_label,
                )

                type_rows.extend(rows)

                if warning:
                    print(f"    {warning}")
                else:
                    print(f"    Collected {len(rows)} rows")

            except Exception as exc:
                failures.append({
                    "institute_type": type_name,
                    "institute_id": institute_value,
                    "institute": institute_label,
                    "error": str(exc),
                })
                print(f"    FAILED: {exc}")

        if type_rows:
            df = pd.DataFrame(type_rows, columns=COLUMNS)
            df = df.drop_duplicates()

            output = OUTPUT_DIR / (
                f"josaa_{year}_round{round_no}_"
                f"{type_name.lower()}_cse_all_categories.csv"
            )
            df.to_csv(output, index=False, encoding="utf-8-sig")
            print(f"\nSaved {len(df)} rows: {output}")

            all_rows.extend(df.to_dict("records"))
        else:
            print(f"No rows collected for {type_name}.")

        if failures:
            failure_file = OUTPUT_DIR / (
                f"josaa_{year}_round{round_no}_"
                f"{type_name.lower()}_failures.csv"
            )
            pd.DataFrame(failures).to_csv(
                failure_file, index=False, encoding="utf-8-sig"
            )
            print(f"Failure log saved: {failure_file}")

    if all_rows:
        combined = pd.DataFrame(all_rows, columns=COLUMNS).drop_duplicates()
        combined_path = OUTPUT_DIR / (
            f"josaa_{year}_round{round_no}_"
            "nit_iiit_gfti_cse_all_categories.csv"
        )
        combined.to_csv(
            combined_path, index=False, encoding="utf-8-sig"
        )

        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total unique rows: {len(combined)}")
        print(combined["institute_type"].value_counts().to_string())
        print(f"\nCombined CSV: {combined_path}")
    else:
        print(
            "\nNo data was collected. Check the error messages above "
            "before trying again."
        )


if __name__ == "__main__":
    main()
