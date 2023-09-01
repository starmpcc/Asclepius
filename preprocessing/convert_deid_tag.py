# Reference: https://www.physionet.org/content/clinical-t5/1.0.0/
import re

DEID_TO_TAG = {
    "First Name": "[Name]",
    "Last Name": "[Name]",
    "Clip Number (Radiology)": "[Reg#]",
    "Hospital": "[Company]",
    "Numeric Identifier": "[Reg#]",
    "Location": "[LOC]",
    "Initials": "[Name]",
    "Known lastname": "[Name]",
    "Known firstname": "[Name]",
    "Age over 90": "[Age]",
    "Medical Record Number": "[Reg#]",
    "Telephone/Fax": "[#]",
    "Name": "[Name]",
    "Street Address": "[LOC]",
    "Date Range": "[DR]",
    "Date range": "[DR]",
    "Serial Number": "[#]",
    "Social Security Number": "[#]",
    "State": "[State]",
    "E-mail address": "[CI]",
    "Company": "[Company]",
    "Pager number": "[#]",
    "Country": "[Country]",
    "MD Number": "[Reg#]",
    "Wardname": "[LOC]",
    "College": "[Company]",
    "Apartment Address": "[LOC]",
    "URL": "[URL]",
    "CC Contact Info": "[CI]",
    "Attending Info": "[CI]",
    "Job Number": "[Reg#]",
    "Year": "[YR]",
    "Month": "[MO]",
    "Day": "[DAY]",
    "Holiday": "[DAY]",
    "Unit Number": "[Reg#]",
    "PO Box": "[LOC]",
    "Dictator Info": "[CI]",
    "Provider Number": "[#]",
}


def convert_deid_to_tag(t):
    for deid in DEID_TO_TAG.keys():
        if deid in t or deid.lower() in t:
            return DEID_TO_TAG[deid]

    if len(t.split("-")) == 3:
        return t

    elif t.strip().isnumeric():
        return "[Reg#]"

    elif t.replace("-", "").replace("/", "").isnumeric():
        return t

    elif "January" in t:
        return t
    elif "February" in t:
        return t
    elif "March" in t:
        return t
    elif "April" in t:
        return t
    elif "May" in t:
        return t
    elif "June" in t:
        return t
    elif "July" in t:
        return t
    elif "August" in t:
        return t
    elif "September" in t:
        return t
    elif "October" in t:
        return t
    elif "November" in t:
        return t
    elif "December" in t:
        return t
    elif "" == t.strip():
        return ""
    else:
        raise NotImplementedError("Not available.")


def get_deid_offsets(text) -> list:
    """Find the offsets for a single note."""
    regex = "\[\*\*(.*?)\*\*\]"

    offsets = []
    for m in re.compile(regex).finditer(text):
        selected_text = m.groups()[0]
        offsets.append((m.start(), m.end(), selected_text))

    return offsets


def replace_list_of_notes(text_to_replace):
    """Take a list of notes and replace the tags with single tokens."""

    all_texts = []
    for text in text_to_replace:
        offsets = get_deid_offsets(text)

        new_text, last_offset = [], 0
        for i, (st, end, span) in enumerate(offsets):
            tag = convert_deid_to_tag(span)
            new_text.append(text[last_offset:st])
            new_text.append(tag)
            last_offset = end

        new_text.append(text[last_offset:])
        all_texts.append(" ".join(filter(lambda x: x != "", new_text)))

    return all_texts
