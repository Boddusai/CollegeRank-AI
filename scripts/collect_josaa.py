import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


URL = "https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx"

session = requests.Session()
session.verify = False

def get_hidden_fields(soup):
    """Get all ASP.NET hidden form fields."""
    data = {}

    for element in soup.select("input[type=hidden]"):
        name = element.get("name")
        value = element.get("value", "")

        if name:
            data[name] = value

    return data


def postback(soup, target, values):
    """Send an ASP.NET dropdown postback."""
    
    form_data = get_hidden_fields(soup)

    form_data["__EVENTTARGET"] = target
    form_data["__EVENTARGUMENT"] = ""
    form_data["__LASTFOCUS"] = ""

    form_data.update(values)

    response = session.post(
        URL,
        data=form_data,
        timeout=30
    )

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )

def get_dropdown_options(soup, dropdown_name):
    """Return all value/text pairs from a dropdown."""

    dropdown = soup.find(
        "select",
        {"name": dropdown_name}
    )

    if not dropdown:
        return []

    options = []

    for option in dropdown.find_all("option"):

        value = option.get("value", "").strip()
        text = option.get_text(
            " ",
            strip=True
        )

        if value and value != "0":
            options.append({
                "value": value,
                "text": text
            })

    return options

def find_branch_by_name(branches, search_text):
    """Find a branch whose name contains the requested text."""

    search_text = search_text.lower()

    for branch in branches:

        if search_text in branch["text"].lower():
            return branch

    return None

def collect_josaa(
    year,
    round_no,
    institute_type,
    institute,
    branch,
    seat_type
):
    print("DEBUG collect_josaa year =", year)
    """
    Collect JoSAA opening and closing rank data
    for one specific combination.
    """

    # -----------------------------------------
    # 1. GET initial page
    # -----------------------------------------

    response = session.get(
        URL,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # -----------------------------------------
    # 2. Select Year
    # -----------------------------------------

    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlYear",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year)
        }
    )

    # -----------------------------------------
    # 3. Select Round
    # -----------------------------------------

    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlroundno",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no)
        }
    )

    # -----------------------------------------
    # 4. Select Institute Type
    # -----------------------------------------

    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlInstype",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
            "ctl00$ContentPlaceHolder1$ddlInstype": institute_type
        }
    )

    # -----------------------------------------
    # 5. Select Institute
    # -----------------------------------------

    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlInstitute",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
            "ctl00$ContentPlaceHolder1$ddlInstype": institute_type,
            "ctl00$ContentPlaceHolder1$ddlInstitute": institute
        }
    )

    # -----------------------------------------
    # 6. Select Branch
    # -----------------------------------------

    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlBranch",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
            "ctl00$ContentPlaceHolder1$ddlInstype": institute_type,
            "ctl00$ContentPlaceHolder1$ddlInstitute": institute,
            "ctl00$ContentPlaceHolder1$ddlBranch": branch
        }
    )

    # -----------------------------------------
    # 7. Final Submit
    # -----------------------------------------

    form_data = get_hidden_fields(soup)

    form_data.update({
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",

        "ctl00$ContentPlaceHolder1$ddlYear": str(year),
        "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
        "ctl00$ContentPlaceHolder1$ddlInstype": institute_type,
        "ctl00$ContentPlaceHolder1$ddlInstitute": institute,
        "ctl00$ContentPlaceHolder1$ddlBranch": branch,
        "ctl00$ContentPlaceHolder1$ddlSeatType": seat_type,

        "ctl00$ContentPlaceHolder1$btnSubmit": "Submit"
    })

    final_response = session.post(
        URL,
        data=form_data,
        timeout=30
    )

    final_response.raise_for_status()

    final_soup = BeautifulSoup(
        final_response.text,
        "html.parser"
    )

    # -----------------------------------------
    # 8. Extract result table
    # -----------------------------------------

    tables = final_soup.find_all("table")

    results = []

    if not tables:
        return results

    # Usually the first table contains results.
    table = tables[0]

    rows = table.find_all("tr")

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])

        values = [
            cell.get_text(
                " ",
                strip=True
            )
            for cell in cells
        ]

        if len(values) >= 7:

            # Ignore completely empty rows
            if not any(values):
                continue

            results.append({
                "year": year,
                "round": round_no,
                "institute_type": institute_type,
                "institute": values[0],
                "academic_program": values[1],
                "quota": values[2],
                "seat_type": values[3],
                "gender": values[4],
                "opening_rank": values[5],
                "closing_rank": values[6]
            })

    return results


# ==================================================
# COLLECT CSE FROM ALL IITs
# ==================================================

import time


def get_iit_list(year, round_no):
    """
    Get the list of IIT institutes available
    for the selected year and round.
    """

    response = session.get(
        URL,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Year
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlYear",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year)
        }
    )

    # Round
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlroundno",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no)
        }
    )

    # Institute Type = IIT
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlInstype",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
            "ctl00$ContentPlaceHolder1$ddlInstype": "IIT"
        }
    )

    institutes = get_dropdown_options(
        soup,
        "ctl00$ContentPlaceHolder1$ddlInstitute"
    )

    # Remove ALL
    institutes = [
        institute
        for institute in institutes
        if institute["value"] != "ALL"
    ]

    return institutes


def get_cse_branch(
    year,
    round_no,
    institute_id
):
    """
    Find the CSE branch code for one IIT.
    """

    response = session.get(
        URL,
        timeout=30
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Year
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlYear",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year)
        }
    )

    # Round
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlroundno",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no)
        }
    )

    # IIT
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlInstype",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
            "ctl00$ContentPlaceHolder1$ddlInstype": "IIT"
        }
    )

    # Institute
    soup = postback(
        soup,
        "ctl00$ContentPlaceHolder1$ddlInstitute",
        {
            "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "ctl00$ContentPlaceHolder1$ddlroundno": str(round_no),
            "ctl00$ContentPlaceHolder1$ddlInstype": "IIT",
            "ctl00$ContentPlaceHolder1$ddlInstitute": institute_id
        }
    )

    branches = get_dropdown_options(
        soup,
        "ctl00$ContentPlaceHolder1$ddlBranch"
    )

    return find_branch_by_name(
        branches,
        "Computer Science and Engineering"
    )


# ==================================================
# MAIN COLLECTION
# ==================================================

if __name__ == "__main__":

    YEAR = int(input("Enter JoSAA year (e.g. 2021): ").strip())
    ROUND = 2
    SEAT_TYPE = "OPNO"

    print()
    print("=" * 70)
    print("JOOSAA IIT CSE COLLECTION")
    print("=" * 70)

    print()
    print("Year:", YEAR)
    print("Round:", ROUND)
    print("Seat Type: OPEN")
    print()

    # -----------------------------------------
    # Get IIT list
    # -----------------------------------------

    institutes = get_iit_list(
        YEAR,
        ROUND
    )

    print(
        "IITs found:",
        len(institutes)
    )

    print()

    all_results = []

    # -----------------------------------------
    # Process each IIT
    # -----------------------------------------

    for index, institute in enumerate(
        institutes,
        start=1
    ):

        institute_id = institute["value"]
        institute_name = institute["text"]

        print(
            f"[{index}/{len(institutes)}] "
            f"{institute_name}"
        )

        try:

            # Find CSE
            cse = get_cse_branch(
                YEAR,
                ROUND,
                institute_id
            )

            if not cse:

                print(
                    "      CSE not found - skipped"
                )

                continue

            print(
                "      CSE found:",
                cse["value"]
            )

            # Collect cutoff
            results = collect_josaa(
                year=YEAR,
                round_no=ROUND,
                institute_type="IIT",
                institute=institute_id,
                branch=cse["value"],
                seat_type=SEAT_TYPE
            )

            print(
                "      Rows collected:",
                len(results)
            )

            all_results.extend(results)

        except Exception as error:

            print(
                "      ERROR:",
                error
            )

        # Small delay between IITs
        time.sleep(1)

    # -----------------------------------------
    # Save results
    # -----------------------------------------

    print()
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)

    print(
        "Total rows collected:",
        len(all_results)
    )

    if all_results:

        df = pd.DataFrame(
            all_results
        )

        output_file = (
            f"data/processed/josaa_{YEAR}_round{ROUND}_iit_cse_open.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print()
        print(
            "CSV saved:",
            output_file
        )

        print()
        print("Preview:")
        print(
            df[
                [
                    "institute",
                    "academic_program",
                    "gender",
                    "opening_rank",
                    "closing_rank"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No data collected."
        )

    print()
    print("=" * 70)