import requests
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd

URL = (
    "https://josaa.admissions.nic.in/"
    "applicant/seatmatrix/openingclosingrankarchieve.aspx"
)

YEAR = "2025"
ROUND = "2"
INSTITUTE_TYPE = "NIT"
INSTITUTE_ID = "201"

YEAR_FIELD = "ctl00$ContentPlaceHolder1$ddlYear"
ROUND_FIELD = "ctl00$ContentPlaceHolder1$ddlroundno"
TYPE_FIELD = "ctl00$ContentPlaceHolder1$ddlInstype"
INSTITUTE_FIELD = "ctl00$ContentPlaceHolder1$ddlInstitute"
BRANCH_FIELD = "ctl00$ContentPlaceHolder1$ddlBranch"
SEAT_TYPE_FIELD = "ctl00$ContentPlaceHolder1$ddlSeatType"


def hidden_fields(soup):
    fields = {}
    for item in soup.select('input[type="hidden"]'):
        name = item.get("name")
        if name:
            fields[name] = item.get("value", "")
    return fields


def postback(session, soup, target, updates):
    data = hidden_fields(soup)
    data.update(updates)
    data["__EVENTTARGET"] = target
    data["__EVENTARGUMENT"] = ""
    data["__LASTFOCUS"] = ""

    response = session.post(URL, data=data, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def show_options(soup, field):
    dropdown = soup.find("select", {"name": field})
    if dropdown is None:
        print("Dropdown not found:", field)
        return

    options = [
        (item.get("value", "").strip(),
         item.get_text(" ", strip=True))
        for item in dropdown.find_all("option")
        if item.get("value", "").strip()
        and item.get("value", "").strip() != "0"
    ]

    print(f"\n{field}: {len(options)} options")
    for value, label in options[:30]:
        print(f"  {value!r} -> {label}")
    if len(options) > 30:
        print(f"  ... and {len(options) - 30} more")

def main():
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        response = session.get(URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        selections = [
            (YEAR_FIELD, YEAR),
            (ROUND_FIELD, ROUND),
            (TYPE_FIELD, INSTITUTE_TYPE),
            (INSTITUTE_FIELD, INSTITUTE_ID),
            (BRANCH_FIELD, "4110"),
        ]

        selected = {}

        for field, value in selections:
            selected[field] = value
            soup = postback(
                session,
                soup,
                field,
                selected.copy(),
            )

        print("\nSelected: NIT Jalandhar")
        print("Branch: Computer Science and Engineering")

        show_options(soup, SEAT_TYPE_FIELD)

        # Submit the selected branch and OPEN category.
        
        selected[SEAT_TYPE_FIELD] = "OPNO"

        # Trigger the official JoSAA Submit button.
        selected["ctl00$ContentPlaceHolder1$btnSubmit"] = "Submit"

        soup = postback(
            session,
            soup,
            "ctl00$ContentPlaceHolder1$btnSubmit",
            selected.copy(),
        )

        print("\nTables found:", len(soup.find_all("table")))
        print("\nResult preview:")
        print(soup.get_text(" ", strip=True)[-5000:])

        
        table = soup.find(
            "table",
            {"id": "ctl00_ContentPlaceHolder1_GridView1"}
        )

        if table is None:
            raise RuntimeError("JoSAA cutoff results table was not found.")

        rows = []

        for tr in table.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in tr.find_all(["th", "td"])
            ]

            if (
                len(cells) == 7
                and cells[0] != "Institute"
                and cells[0].strip()
                and cells[1].strip()
                and cells[5].strip()
                and cells[6].strip()
            ):
                rows.append(cells)

        columns = [
            "institute",
            "academic_program",
            "quota",
            "seat_type",
            "gender",
            "opening_rank",
            "closing_rank",
        ]

        df = pd.DataFrame(rows, columns=columns)


        output = Path("data/raw/jee_main")
        output.mkdir(parents=True, exist_ok=True)

        csv_path = output / "josaa_2025_round2_nit_jalandhar_cse_open.csv"
        df.to_csv(csv_path, index=False)

        print("\nParsed cutoff rows:", len(df))
        print(df.to_string(index=False))
        print("\nSaved to:", csv_path)


        print("\nPage title:", soup.title.get_text(" ", strip=True) if soup.title else "No title")
        
        print("\n--- BUTTONS ---")
        for item in soup.find_all(["input", "button", "a"]):
            label = (
                item.get("value")
                or item.get_text(" ", strip=True)
                or item.get("title")
                or ""
            ).strip()

            if label:
                print(
                    "tag =", item.name,
                    "| text =", repr(label),
                    "| name =", item.get("name"),
                    "| id =", item.get("id"),
                    "| href =", item.get("href")
                )

        print("\n--- FORM DETAILS ---")
        for form in soup.find_all("form"):
            print("Form ID:", form.get("id"))
            print("Form action:", form.get("action"))

        print("Final URL:", response.url)

        print("\nPage text preview:")
        print(soup.get_text(" ", strip=True)[:5000])

        print("\nElements containing rank/cutoff:")
        for element in soup.find_all(["div", "span", "table"]):
            text = element.get_text(" ", strip=True)
            if any(word in text.lower() for word in ["opening rank", "closing rank", "opening/closing", "no records"]):
                print(element.name, element.get("id"), repr(text[:500]))


if __name__ == "__main__":
    main()
