
import os
import pandas as pd
from lxml import etree

def parse_article(file_content):
    """
    Parses the content of a single XML file representing an article.
    Extracts headline and combines all text content under <body> into one column.
    """
    def extract_text_recursive(element):
        """
        Recursively extracts all text and tail content under an element.
        """
        text_content = []
        if element.text:
            text_content.append(element.text.strip())  # Extract text content of the current tag
        for child in element:
            text_content.append(extract_text_recursive(child))  # Recurse into child tags
            if child.tail:
                text_content.append(child.tail.strip())  # Extract tail text after a child
        return " ".join(filter(None, text_content))  # Join non-empty content

    try:
        root = etree.fromstring(file_content.encode("ISO-8859-1"))

        # Extract attributes from <doc>
        doc_attributes = root.attrib

        # Extract attributes from <djnml>
        djnml = root.find("djnml")
        djnml_attributes = djnml.attrib if djnml is not None else {}

        # Extract data from <djn-mdata>
        djn_mdata = djnml.find(".//djn-mdata") if djnml is not None else None
        accession_number = djn_mdata.attrib.get("accession-number", "Unknown") if djn_mdata is not None else "Unknown"
        display_date = djn_mdata.attrib.get("display-date", "Unknown") if djn_mdata is not None else "Unknown"
        orig_source = djn_mdata.attrib.get("original-source", "Unknown") if djn_mdata is not None else "Unknown"

        # Extract data from <head>
        #head = djnml.find("head") if djnml is not None else None
        #copyright_year = head.find("copyright").attrib.get("year", "Unknown") if head is not None else "Unknown"
        #copyright_holder = head.find("copyright").attrib.get("holder", "Unknown") if head is not None else "Unknown"

        # Extract coding data
        coding = djnml.find(".//djn-coding") if djnml is not None else None
        companies = [c.text for c in coding.findall(".//djn-company/c")] if coding is not None else []
        industries = [c.text for c in coding.findall(".//djn-industry/c")] if coding is not None else []
        subjects = [c.text for c in coding.findall(".//djn-subject/c")] if coding is not None else []
        markets = [c.text for c in coding.findall(".//djn-market/c")] if coding is not None else []
        products = [c.text for c in coding.findall(".//djn-product/c")] if coding is not None else []
        geos = [c.text for c in coding.findall(".//djn-geo/c")] if coding is not None else []

        # Extract data from <body>
        body = djnml.find("body") if djnml is not None else None

        # Extract headline
        headline = body.findtext("headline", default="Unknown Headline") if body is not None else "Unknown Headline"

        # Extract all text content under <body>, regardless of tags
        body_content = extract_text_recursive(body) if body is not None else ""

        return {
            "Doc Attributes": doc_attributes,
            "Djnml Attributes": djnml_attributes,
            "Accession Number": accession_number,
            "Original Source": orig_source,
            "Display Date": display_date,
            #"Copyright Year": copyright_year,
            #"Copyright Holder": copyright_holder,
            "Companies": companies,
            "Industries": industries,
            "Subjects": subjects,
            "Markets": markets,
            "Products": products,
            "Geos": geos,
            "Headline": headline,
            "Body Text": body_content  # Combined text under <body>, regardless of tags
        }
    except etree.XMLSyntaxError as e:
        print(f"Error parsing article: {e}")
        return None


def parse_article_new(file_content):
    """
    Parses the content of a single XML file representing an article.
    Extracts all tags, attributes, and text dynamically into a flat dictionary.
    """

    def flatten_element(element, parent_key=""):
        """
        Recursively flattens an XML element into a dictionary.
        """
        flat_data = {}
        current_key = f"{parent_key}/{element.tag}" if parent_key else element.tag

        # Include attributes of the current tag
        for attr_key, attr_value in element.attrib.items():
            flat_data[f"{current_key}@{attr_key}"] = attr_value

        # Include text content if present
        if element.text and element.text.strip():
            flat_data[f"{current_key}@text"] = element.text.strip()

        # Recurse into children
        for child in element:
            child_data = flatten_element(child, current_key)
            flat_data.update(child_data)

        return flat_data

    try:
        # Parse the XML content
        root = etree.fromstring(file_content.encode("ISO-8859-1"))
        flat_data = flatten_element(root)

        # Aggregate all paragraphs and preformatted text into "body_text"
        paragraphs = []
        for key, value in flat_data.items():
            if key.endswith("body/text/p@text") or key.endswith("body/text/pre@text"):
                paragraphs.append(value)

        flat_data["body_text"] = "\n".join(paragraphs)

        return flat_data

    except etree.XMLSyntaxError as e:
        print(f"Error parsing article: {e}")
        return None


def process_nml_file(file_path):
    articles = []
    with open(file_path, 'r', encoding="ISO-8859-1") as f:
        file_content = ""
        for line in f:
            if line.strip().startswith('<?xml'):
                # If a new article starts, process the current one
                if file_content.strip():
                    article_data = parse_article(file_content)
                    if article_data:
                        articles.append(article_data)
                    file_content = ""  # Reset the content for the next article
            file_content += line
        if file_content.strip():
            article_data = parse_article(file_content)
            if article_data:
                articles.append(article_data)

    # Create a DataFrame from the parsed articles
    df = pd.DataFrame(articles)
    doc_attributes_expanded = df['Doc Attributes'].apply(pd.Series)
    djnml_attributes_expanded = df['Djnml Attributes'].apply(pd.Series)
    df_expanded = pd.concat([df.drop(['Doc Attributes', 'Djnml Attributes'], axis=1),
                             doc_attributes_expanded, djnml_attributes_expanded], axis=1)
    df_expanded['Date'] = pd.to_datetime(df_expanded['docdate'], format='%Y%m%d')
    df_expanded['Headline'] = df_expanded['Headline'].str.replace('\n', ' ', regex=False)
    df_expanded['Body Text'] = df_expanded['Body Text'].str.replace('\n', ' ', regex=False)

    return df_expanded


def simplify_column_names(df):
    new_columns = {}
    for col in df.columns:
        simplified = col.replace('doc/', '').replace('djnml/', '').replace('head/', '')
        simplified = simplified.replace('docdata/', '').replace('djn-newswires/', '')
        simplified = simplified.replace('djn-', '').replace('coding/', '')
        simplified = simplified.replace('@', '_')
        new_columns[col] = simplified.strip('.')
    df.rename(columns=new_columns, inplace=True)
    return df


def process_nml_file_new(file_path):
    articles = []
    with open(file_path, 'r', encoding="ISO-8859-1") as f:
        file_content = ""
        for line in f:
            if line.strip().startswith('<?xml'):
                if file_content.strip():
                    article_data = parse_article_new(file_content)
                    if article_data:
                        articles.append(article_data)
                    file_content = ""
            file_content += line
        if file_content.strip():
            article_data = parse_article_new(file_content)
            if article_data:
                articles.append(article_data)

    # Create a DataFrame from the parsed articles
    df = pd.DataFrame(articles)

    # Simplify column names
    df = simplify_column_names(df)

    if 'djnml_docdate' in df.columns:
        df['Date'] = pd.to_datetime(df['djnml_docdate'], format='%Y%m%d', errors='coerce')
    if 'body_headline_text' in df.columns:
        df['Headline'] = df['body_headline_text'].str.replace('\n', ' ', regex=False)
    if 'body_text' in df.columns:
        df['Body Text'] = df['body_text'].str.replace('\n', ' ', regex=False)

    return df

if __name__ == "__main__":
    data = process_nml_file("/Users/lukavulicevic/Desktop/Data Storage/DJ_Archive_2000_2023/Raw Files/2023-01.nml")
    #data.to_csv("/Volumes/LV SSD/DJ_Archive_2000_2023/DataFrames/2009-01.csv")

    for y in range(2024, 2025):
        for m in range(1, 13):
            if m < 10:
                month = '0' + str(m)
            else:
                month = str(m)
            year = str(y)
            print("Currently on: " + year + '-' + month, end='\r')
            try:
                data = process_nml_file(f"/Users/lukavulicevic/Desktop/Data Storage/DJ_Archive_2000_2023/Raw Files/{year}-{month}.nml")
                data.to_csv(f"/Users/lukavulicevic/Desktop/Data Storage/DJ_Archive_2000_2023/BY Month/{year}-{month}.csv")
            except Exception as e:
                print(f"\nError occurred for {y}-{m}: {e}")
                continue

    for y in range(2022, 2025):
        dataframes = []
        year = str(y)
        for m in range(1, 13):
            if m < 10:
                month = '0' + str(m)
            else:
                month = str(m)
            print("Currently on: " + year + '-' + month, end='\r')
            try:
                data = pd.read_csv(f"/Users/lukavulicevic/Desktop/Data Storage/DJ_Archive_2000_2023/BY Month/{year}-{month}.csv",low_memory=False)
                dataframes.append(data)
            except Exception as e:
                print(f"\nError occurred for {y}-{m}: {e}")
                continue

        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv("/Users/lukavulicevic/Desktop/Data Storage/DJ_Archive_2000_2023/DataFrames/Year/" + year + ".csv")

    dataframes = []
    for y in range(2016, 2024):
        year = str(y)
        print("Currently on: " + year, end='\r')
        try:
            data = pd.read_csv(f"/Users/lukavulicevic/Desktop/Data Storage/DJ_Archive_2000_2023/DataFrames/Year/{year}.csv")
        except Exception as e:
            print(f"\nError occurred for {y}: {e}")
            continue
        dataframes.append(data)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv("/Volumes/LV SSD/DJ_Archive_2000_2023/DataFrames/Decades/2016_2023.csv")


