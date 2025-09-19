import fitz  # PyMuPDF
import re
import json
import os

def parse_mas_notice(pdf_path):
    """
    Parses a MAS circular PDF, chunking it into paragraphs and sub-paragraphs.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A JSON string representing the structured data.
    """
    filename = os.path.basename(pdf_path)

    # 1. Extract metadata from filename
    metadata = {}
    match = re.search(r'MAS Notice (\w+)_dated ([\d\w\s]+)_effective ([\d\w\s]+)\.pdf', filename)
    if match:
        metadata['notice_id'] = f"MAS Notice {match.group(1)}"
        metadata['publication_date'] = match.group(2)
        metadata['effective_date'] = match.group(3)
    else:
        metadata['notice_id'] = "Unknown"
        metadata['publication_date'] = "Unknown"
        metadata['effective_date'] = "Unknown"

    doc = fitz.open(pdf_path)

    nodes = []
    node_counter = 1
    last_top_level_node = None

    # Regex patterns
    para_num_pattern = re.compile(r'^(\d+[A-Z]?)\.?\s+.*') # Starts with "1." or "1A "
    para_alpha_pattern = re.compile(r'^\(([a-z])\)\s+.*') # Starts with "(a) "

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    doc.close()

    # Split the text into lines, which is a simpler approach than blocks
    lines = full_text.split('\n')

    current_node_text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_new_para = para_num_pattern.match(line)
        is_new_sub_para = para_alpha_pattern.match(line)

        # If we find a new paragraph or sub-paragraph marker,
        # we save the previously accumulated text as a node.
        if is_new_para or is_new_sub_para:
            # First, save the previous node if it exists
            if nodes:
                nodes[-1]['text'] = re.sub(r'\s+', ' ', nodes[-1]['text']).strip()

            # Now, create the new node
            parent_id = None
            node_type = "paragraph"
            if is_new_sub_para:
                node_type = "sub-paragraph"
                if last_top_level_node:
                    parent_id = last_top_level_node['node_id']

            node_id = f"node_{node_counter}"
            new_node = {
                "node_id": node_id,
                "node_type": node_type,
                "text": line,
                "parent_id": parent_id,
                "metadata": {"source_filename": filename}
            }
            nodes.append(new_node)
            node_counter += 1

            if is_new_para:
                last_top_level_node = new_node

        elif nodes:
            # This is a continuation line, append it to the last node
            nodes[-1]['text'] += f" {line}"

    # Final cleanup for the very last node
    if nodes:
        nodes[-1]['text'] = re.sub(r'\s+', ' ', nodes[-1]['text']).strip()


    # A fallback for documents that don't match the paragraph structure
    if not nodes:
        nodes.append({
            "node_id": "node_1",
            "node_type": "full_text",
            "text": re.sub(r'\s+', ' ', full_text).strip(),
            "parent_id": None,
            "metadata": {"source_filename": filename}
        })


    structured_data = {
        "metadata": metadata,
        "content": nodes
    }

    return json.dumps(structured_data, indent=4)

if __name__ == '__main__':
    pdf_file = 'data/MAS Notice 758_dated 18 Dec 2024_effective 26 Dec 2024.pdf'
    if os.path.exists(pdf_file):
        json_output = parse_mas_notice(pdf_file)

        output_filename = f"data/{os.path.basename(pdf_file).replace('.pdf', '_structured.json')}"
        with open(output_filename, 'w') as f:
            f.write(json_output)

        print(f"Parsed data saved to {output_filename}")
    else:
        print(f"Error: PDF file not found at {pdf_file}")
