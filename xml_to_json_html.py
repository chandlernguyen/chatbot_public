#!/usr/bin/env python3

import os
import re
from lxml import etree
from bs4 import BeautifulSoup
import json

def generate_post_content(html_content):
    """
    Cleans the given HTML content by removing script and style tags.
    
    Args:
        html_content (str): The raw HTML content to clean.
    
    Returns:
        str: The cleaned HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()  # Remove these tags and their contents
    return str(soup)

def sanitize_filename(filename):
    """
    Sanitizes the filename by removing or replacing specific characters not suitable for filenames.
    
    Args:
        filename (str): The original filename to sanitize.
    
    Returns:
        str: The sanitized filename.
    """
    sanitized = re.sub(r'[^\w\s-]', '', filename).strip().lower()
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized

# Define a list of known categories from the website
# Replace the example categories with your actual categories
KNOWN_CATEGORIES = [
    "Category 1", "Category 2", "Category 3",  # Add or replace with your own categories
]

# Initialize an empty string to hold all HTML content
all_posts_html = ""
all_posts = []

try:
    # Parse the XML file exported from WordPress
    # Replace 'your_export_file.xml' with the name of your actual WordPress export file
    tree = etree.parse('your_export_file.xml')
    root = tree.getroot()

    # Define namespace used in the WordPress export file
    namespaces = {'content': 'http://purl.org/rss/1.0/modules/content/'}

    # Prompt user for output directory or use a default
    output_folder = input("Enter the desired output folder (default: html_outputs): ") or "html_outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each post in the export file
    for item in root.findall(".//item"):
        post_title = item.find("title").text
        post_content = item.find("content:encoded", namespaces=namespaces).text
        pub_date = item.find("pubDate").text
        categories_and_tags = [cat.text for cat in item.findall("category")]

        # Separate categories and tags
        categories = [entry for entry in categories_and_tags if entry in KNOWN_CATEGORIES]
        tags = [entry for entry in categories_and_tags if entry not in KNOWN_CATEGORIES]

        post_link = item.find("link").text

        clean_content = generate_post_content(post_content)
        post_dict = {
            "title": post_title,
            "published_date": pub_date,
            "url": post_link,
            "categories": categories,
            "tags": tags,
            "content": clean_content
        }

        all_posts.append(post_dict)

        enhanced_content = f"""
Published on: {pub_date}
URL: {post_link}
Categories: {', '.join(categories)}
Tags: {', '.join(tags)}

<!-- Start of Content -->
{clean_content}
<!-- End of Content -->
"""

        sanitized_title = sanitize_filename(post_title)
        with open(os.path.join(output_folder, f"{sanitized_title}.html"), "w", encoding="utf-8") as f:
            f.write(enhanced_content)

        all_posts_html += enhanced_content

    print("HTML files generation completed!")

    # Convert and save all posts as a JSON file
    with open(f"{output_folder}/all_posts.json", "w") as json_file:
        json.dump({"posts": all_posts}, json_file)

except Exception as e:
    print(f"An error occurred: {e}")

# Write all posts to a single HTML file for convenience
with open(f"{output_folder}/all_posts.html", "w") as f:
    f.write(all_posts_html)
