import requests
from bs4 import BeautifulSoup


URL = "https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx"

session = requests.Session()


def get_hidden_fields(soup):
    data = {}

    for element in soup.select("input[type=hidden]"):
        name = element.get("name")
        value = element.get("value", "")

        if name:
            data[name] = value

    return data


def postback(soup, target, values):
    form_data = get_hidden_fields(soup)

    form_data["__EVENTTARGET"] = target
    form_data["__EVENTARGUMENT"] = ""
    form_data["__LASTFOCUS"] = ""

    form_data.update(values)

    return session.post(
        URL,
        data=form_data,
        timeout=30
    )


# ==================================================
# 1. GET
# ==================================================

response = session.get(URL, timeout=30)

print("GET status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


# ==================================================
# 2. YEAR = 2025
# ==================================================

response = postback(
    soup,
    "ctl00$ContentPlaceHolder1$ddlYear",
    {
        "ctl00$ContentPlaceHolder1$ddlYear": "2025"
    }
)

print("Year POST status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


# ==================================================
# 3. ROUND = 2
# ==================================================

response = postback(
    soup,
    "ctl00$ContentPlaceHolder1$ddlroundno",
    {
        "ctl00$ContentPlaceHolder1$ddlYear": "2025",
        "ctl00$ContentPlaceHolder1$ddlroundno": "2"
    }
)

print("Round POST status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


# ==================================================
# 4. INSTITUTE TYPE = IIT
# ==================================================

response = postback(
    soup,
    "ctl00$ContentPlaceHolder1$ddlInstype",
    {
        "ctl00$ContentPlaceHolder1$ddlYear": "2025",
        "ctl00$ContentPlaceHolder1$ddlroundno": "2",
        "ctl00$ContentPlaceHolder1$ddlInstype": "IIT"
    }
)

print("IIT POST status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


# ==================================================
# 5. INSTITUTE = IIT HYDERABAD
# ==================================================

response = postback(
    soup,
    "ctl00$ContentPlaceHolder1$ddlInstitute",
    {
        "ctl00$ContentPlaceHolder1$ddlYear": "2025",
        "ctl00$ContentPlaceHolder1$ddlroundno": "2",
        "ctl00$ContentPlaceHolder1$ddlInstype": "IIT",
        "ctl00$ContentPlaceHolder1$ddlInstitute": "107"
    }
)

print("Institute POST status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


# ==================================================
# 6. BRANCH = CSE
# ==================================================

response = postback(
    soup,
    "ctl00$ContentPlaceHolder1$ddlBranch",
    {
        "ctl00$ContentPlaceHolder1$ddlYear": "2025",
        "ctl00$ContentPlaceHolder1$ddlroundno": "2",
        "ctl00$ContentPlaceHolder1$ddlInstype": "IIT",
        "ctl00$ContentPlaceHolder1$ddlInstitute": "107",
        "ctl00$ContentPlaceHolder1$ddlBranch": "4110"
    }
)

print("Branch POST status:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")


# ==================================================
# 7. FINAL SUBMIT
# ==================================================

form_data = get_hidden_fields(soup)

form_data.update({
    "__EVENTTARGET": "",
    "__EVENTARGUMENT": "",
    "__LASTFOCUS": "",

    "ctl00$ContentPlaceHolder1$ddlYear": "2025",
    "ctl00$ContentPlaceHolder1$ddlroundno": "2",
    "ctl00$ContentPlaceHolder1$ddlInstype": "IIT",
    "ctl00$ContentPlaceHolder1$ddlInstitute": "107",
    "ctl00$ContentPlaceHolder1$ddlBranch": "4110",
    "ctl00$ContentPlaceHolder1$ddlSeatType": "OPNO",

    "ctl00$ContentPlaceHolder1$btnSubmit": "Submit"
})


print("\nSending FINAL SUBMIT...")

final_response = session.post(
    URL,
    data=form_data,
    timeout=30
)

print("Final POST status:", final_response.status_code)
print("Final response size:", len(final_response.text))


# ==================================================
# 8. FIND TABLES
# ==================================================

final_soup = BeautifulSoup(
    final_response.text,
    "html.parser"
)

tables = final_soup.find_all("table")

print("\nNumber of tables:", len(tables))


# ==================================================
# 9. PRINT TABLE DATA
# ==================================================

if not tables:

    print("\nNO TABLE FOUND.")

    page_text = final_soup.get_text(
        "\n",
        strip=True
    )

    print("\nPAGE TEXT:")
    print(page_text[:5000])

else:

    print("\n========== JOOSAA RESULT ==========\n")

    for table_number, table in enumerate(tables, start=1):

        print(f"\nTABLE {table_number}")
        print("----------------------------------------")

        for row in table.find_all("tr"):

            cells = row.find_all(["th", "td"])

            values = [
                cell.get_text(
                    " ",
                    strip=True
                )
                for cell in cells
            ]

            if values:
                print(" | ".join(values))


# ==================================================
# 10. SAVE FINAL RESPONSE
# ==================================================

with open(
    "scripts\\josaa_final_response.html",
    "w",
    encoding="utf-8"
) as file:

    file.write(final_response.text)


print("\nSaved:")
print("scripts\\josaa_final_response.html")