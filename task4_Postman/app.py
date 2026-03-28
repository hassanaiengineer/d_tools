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

# Global variable to store parsed templates
parsed_templates = []

def sanitize_variable_name(var_name):
    """
    Sanitizes variable names to be used as regex group names.
    Converts to lowercase and replaces spaces with underscores.
    Removes any non-alphanumeric characters.
    """
    sanitized = re.sub(r'\W+', '_', var_name.lower())
    return sanitized

def process_variable(var):
    """
    Processes the extracted variable.
    If the variable contains a comma, splits it into a list.
    Otherwise, returns the variable as a string.
    """
    if isinstance(var, str):
        var = var.strip()
        if ',' in var:
            return [v.strip() for v in var.split(',')]
    return var

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
    # Handle multiple occurrences by making group names unique
    var_pattern = re.compile(r'\\\{(\w+)\\\}')
    var_matches = var_pattern.findall(escaped_template)
    var_count = {}
    for var in var_matches:
        count = var_count.get(var, 0) + 1
        var_count[var] = count
        group_name = f"{sanitize_variable_name(var)}_{count}"
        # Define specific patterns based on variable names if needed
        if var.lower() in ['start_date', 'end_date', 'date']:
            group_pattern = rf'(?P<{group_name}>\\d{{4}}-\\d{{2}}-\\d{{2}})'
        elif var.lower() in ['start_time', 'end_time', 'time']:
            group_pattern = rf'(?P<{group_name}>\\d{{1,2}}:\\d{{2}}(?:\\s?[APMapm]{2})?)'
        elif var.lower() in ['admission_number', 'number']:
            group_pattern = rf'(?P<{group_name}>\\d+)'
        elif var.lower() in ['percentage', 'percent']:
            group_pattern = rf'(?P<{group_name}>\\d+%?)'
        else:
            # For general variables, capture until the next placeholder or end
            group_pattern = rf'(?P<{group_name}>.+?)'
        
        # Replace only the first occurrence each time
        escaped_template = var_pattern.sub(group_pattern, escaped_template, count=1)
        logger.debug(f"Replaced '{{{var}}}' with '{group_pattern}'")
    
    # Replace [option1/option2/...]
    escaped_template = re.sub(
        r'\\\[(.*?)\\\]',
        lambda m: f"(?:{'|'.join([opt.strip() for opt in re.split(r'[\/,]', m.group(1))])})",
        escaped_template
    )
    logger.debug(f"After replacing [options]: {escaped_template}")

    # Replace <option1, option2, ...>
    escaped_template = re.sub(
        r'\\\<([^\\]+)\\\>',
        lambda m: f"(?:{'|'.join([opt.strip() for opt in re.split(r'[\/,]', m.group(1))])})",
        escaped_template
    )
    logger.debug(f"After replacing <options>: {escaped_template}")

    # Anchor the pattern to match the entire string
    final_pattern = f'^{escaped_template}$'
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
            data = json.load(f)
            templates = data.get('questions', [])
            logger.info(f"Loaded {len(templates)} templates from templates.json")
    except FileNotFoundError:
        logger.error("templates.json file not found.")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding templates.json: {e}")
        return []
    
    parsed = []
    for template in templates:
        regex_pattern = generate_regex_from_template(template)
        try:
            compiled_regex = re.compile(regex_pattern, re.IGNORECASE)  # Case-insensitive matching
            parsed.append({
                'template': template,
                'regex': compiled_regex
            })
            logger.debug(f"Compiled regex for template: '{template}' => '{regex_pattern}'")
        except re.error as e:
            logger.error(f"Invalid regex pattern for template '{template}': {e}")
    
    logger.info(f"Parsed {len(parsed)} templates.")
    return parsed

def fill_template(template, variables):
    """
    Fills the template with the extracted variables.
    Replaces placeholders with actual values enclosed in angle brackets.
    If a variable is a list, joins its elements with commas.
    
    Args:
        template (str): The original template string.
        variables (dict): Extracted variables from the query.
    
    Returns:
        str: The filled template with actual values.
    """
    filled = template
    for var, value in variables.items():
        # Extract the base variable name by removing the count suffix
        base_var = var.rsplit('_', 1)[0]
        if isinstance(value, list):
            value_str = ', '.join(value)
        else:
            value_str = value
        # Replace all occurrences of the base variable placeholder
        filled = re.sub(rf'\{{{base_var}\}}|<{base_var}>|\[{base_var}\]', f'<{value_str}>', filled)
    return filled

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
def process_query_route():
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
            filled_template = fill_template(template['template'], variables)
            logger.info(f"Matched Template: {template['template']}")
            logger.info(f"Filled Template: {filled_template}")
            logger.info(f"Extracted Variables: {variables}")
            return jsonify({
                "matched_template": filled_template,
                "variables": variables
            }), 200

    logger.warning("No matching template found for the query.")
    return jsonify({"error": "No matching template found."}), 404

@app.route('/refresh_templates', methods=['GET'])
def refresh_templates():
    """
    Endpoint to refresh templates by reloading 'templates.json'.
    Useful if 'templates.json' is updated without restarting the server.
    
    Returns:
        JSON response indicating success or failure.
    """
    parsed = load_parsed_templates()
    if parsed:
        logger.info("Templates refreshed successfully.")
        return jsonify({"message": "Templates refreshed successfully."}), 200
    else:
        logger.error("Failed to refresh templates.")
        return jsonify({"error": "Failed to refresh templates."}), 500

if __name__ == '__main__':
    parsed_templates = load_parsed_templates()
    app.run(debug=True)
