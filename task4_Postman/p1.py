import re
import json
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s %(levelname)s:%(message)s'
)

logger = logging.getLogger(__name__)

def generate_regex_from_template(template):
    """
    Convert a template string with placeholders into a regex pattern.
    Handles three types of placeholders:
    1. {placeholder}: Named capturing groups.
    2. [option1/option2/...]: Non-capturing group with alternatives.
    3. <option1, option2, ...>: Non-capturing group with alternatives.
    
    Args:
        template (str): The template string with placeholders.
    
    Returns:
        str: The regex pattern string.
    """
    # Escape special regex characters in the template
    escaped_template = re.escape(template)
    logger.debug(f"Escaped template: {escaped_template}")

    # Replace {placeholder} with named capturing groups
    pattern = re.sub(r'\\\{(\w+)\\\}', r'(?P<\1>.+)', escaped_template)
    logger.debug(f"After replacing {{placeholder}}: {pattern}")

    # Function to replace square brackets [option1/option2/...]
    def replace_square_brackets(match):
        # Split options by '/' or ',' and join with '|'
        options = [opt.strip() for opt in re.split(r'[\/,]', match.group(1))]
        regex_options = '|'.join(options)
        logger.debug(f"Replacing [ {match.group(1)} ] with (?:{regex_options})")
        return f"(?:{regex_options})"

    # Function to replace angle brackets <option1, option2, ...>
    def replace_angle_brackets(match):
        options = [opt.strip() for opt in re.split(r'[\/,]', match.group(1))]
        regex_options = '|'.join(options)
        logger.debug(f"Replacing < {match.group(1)} > with (?:{regex_options})")
        return f"(?:{regex_options})"

    # Replace [options] with regex alternatives
    pattern = re.sub(r'\\\[(.*?)\\\]', replace_square_brackets, pattern)
    logger.debug(f"After replacing [options]: {pattern}")

    # Replace <options> with regex alternatives
    pattern = re.sub(r'\\\<([^\\]+)\\\>', replace_angle_brackets, pattern)
    logger.debug(f"After replacing <options>: {pattern}")

    # Anchor the pattern to match the entire string
    final_pattern = f'^{pattern}$'
    logger.debug(f"Final regex pattern: {final_pattern}")

    return final_pattern

def load_parsed_templates():
    """
    Load templates from 'templates.json', convert them into regex patterns,
    and compile them for matching.
    
    Returns:
        list: A list of dictionaries with 'template' and 'regex' keys.
    """
    try:
        with open('templates.json', 'r', encoding='utf-8') as f:
            templates = json.load(f)['questions']  # Use the "questions" key from your JSON
            logger.info(f"Loaded {len(templates)} templates from templates.json")
    except FileNotFoundError:
        logger.error("templates.json file not found.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding templates.json: {e}")
        return []
    
    parsed_templates = []
    for template in templates:
        regex_pattern = generate_regex_from_template(template)
        try:
            compiled_regex = re.compile(regex_pattern, re.IGNORECASE)  # Case-insensitive matching
            parsed_templates.append({
                'template': template,
                'regex': compiled_regex
            })
            logger.debug(f"Compiled regex for template: '{template}' => '{regex_pattern}'")
        except re.error as e:
            logger.error(f"Invalid regex pattern for template '{template}': {e}")
    
    logger.info(f"Parsed {len(parsed_templates)} templates.")
    return parsed_templates

# Initialize templates
parsed_templates = load_parsed_templates()

@app.route('/')
def home():
    """
    Default homepage route to avoid 404 error. Provides basic information
    and allows for interactive query testing.
    """
    return '''
    <h1>Welcome to the Query Processing API</h1>
    <p>Use the <strong>/process_query</strong> POST endpoint to send queries.</p>
    <p>Example template queries:</p>
    <ul>
        <li>"Give me the list of teachers teaching in class 1a from 9 AM to 10 AM on Monday"</li>
        <li>"Show me the attendance record for admission number 67890 from 2024-01-01 to 2024-02-01"</li>
        <li>"Which students in class 6A have less than 75% attendance?"</li>
    </ul>
    '''

@app.route('/process_query', methods=['POST'])
def process_query():
    """
    Endpoint to process incoming queries. Matches the query against predefined
    templates and extracts variables if a match is found.
    
    Expects a JSON payload with a 'query' key.
    
    Returns:
        JSON response with matched template and extracted variables,
        or an error message if no match is found.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        logger.warning("No query provided in the request.")
        return jsonify({"error": "No query provided."}), 400
    
    query = data['query']
    logger.info(f"Received Query: {query}")
    
    for template in parsed_templates:
        logger.debug(f"Trying to match with template: {template['template']}")
        logger.debug(f"Regex pattern: {template['regex'].pattern}")
        logger.debug(f"User query: {query}")  # Log the query you're trying to match

        match = template['regex'].match(query)
        if match:
            variables = match.groupdict()
            filled_template = template['template']
            for key, value in variables.items():
                filled_template = filled_template.replace(f"{{{key}}}", f"<{value}>")
            logger.info(f"Matched Template: {template['template']}")
            logger.info(f"Filled Template: {filled_template}")
            logger.info(f"Extracted Variables: {variables}")
            return jsonify({
                "matched_template": filled_template,
                "variables": variables
            }), 200

    
    logger.warning("No matching template found for the query.")
    return jsonify({"error": "No matching template found."}), 404

if __name__ == '__main__':
    app.run(debug=True)
