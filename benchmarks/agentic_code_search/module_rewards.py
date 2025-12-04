import logging
from typing import Dict, List, Tuple


def parse_simple_output(raw_output: str) -> List[Dict[str, str]]:
    """
    Parse simplified agent output containing filename, optional class, and function.

    Args:
        raw_output: Raw text output from the agent

    Returns:
        List of dictionaries with keys: 'file', 'class' (optional), 'function'

    Example input format:
        ```
        path/to/file1.py
        class: MyClass
        function: my_method

        path/to/file2.py
        function: standalone_function
        ```

    Example output:
        [
            {'file': 'path/to/file1.py', 'class': 'MyClass', 'function': 'my_method'},
            {'file': 'path/to/file2.py', 'class': None, 'function': 'standalone_function'}
        ]
    """
    # Remove triple backticks and whitespace
    raw_output = raw_output.strip("` \n")

    locations = []
    current_file = None
    current_class = None

    lines = raw_output.strip().split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            # Empty line resets the current class context
            current_class = None
            continue

        # Check if this is a Python file path
        if line.endswith(".py"):
            current_file = line
            current_class = None
            continue

        # Parse class declaration
        if line.startswith("class:"):
            class_name = line[len("class:") :].strip()
            current_class = class_name
            continue

        # Parse function/method declaration
        if line.startswith("function:") or line.startswith("method:"):
            if not current_file:
                logging.warning(f"Found function/method without a file: {line}")
                continue

            func_text = line.split(":", 1)[1].strip()
            func_name = func_text.split()[0].strip("() ")

            # Check if function includes class prefix (e.g., "MyClass.my_method")
            if "." in func_name:
                parts = func_name.split(".", 1)
                class_name = parts[0]
                method_name = parts[1]

                locations.append(
                    {"file": current_file, "class": class_name, "function": method_name}
                )
            else:
                # Standalone function or method within current class context
                locations.append(
                    {
                        "file": current_file,
                        "class": current_class,
                        "function": func_name,
                    }
                )

    return locations


def convert_to_entity_format(locations: List[Dict[str, str]]) -> List[str]:
    """
    Convert location dictionaries to entity identifier format.

    Args:
        locations: List of dicts with 'file', 'class', 'function' keys

    Returns:
        List of entity identifiers in format 'file.py:ClassName.function_name'
        or 'file.py:function_name' for standalone functions

    Example:
        Input: [{'file': 'test.py', 'class': 'MyClass', 'function': 'method'}]
        Output: ['test.py:MyClass.method']
    """
    entities = []

    for loc in locations:
        file_path = loc["file"]
        class_name = loc.get("class")
        func_name = loc["function"]

        if class_name:
            entity = f"{file_path}:{class_name}.{func_name}"
        else:
            entity = f"{file_path}:{func_name}"
        if entity.endswith(".__init__"):
            entity = entity[: (len(entity) - len(".__init__"))]
        entities.append(entity)
    entities = list(set(entities))  # Remove duplicates
    return entities


def get_simple_results_from_raw_outputs(
    raw_output: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Process raw output and extract files, modules, and entities.

    This is a simplified version of get_loc_results_from_raw_outputs() that
    doesn't require a dependency graph for validation.

    Args:
        raw_output: Raw text output from the agent

    Returns:
        Tuple of (all_found_files, all_found_modules, all_found_entities)
        where each is a list of strs
    """
    all_found_files = []
    all_found_modules = []
    all_found_entities = []

    locations = parse_simple_output(raw_output)
    files = list(set([loc["file"] for loc in locations]))
    # Convert to entity format
    entities = convert_to_entity_format(locations)

    # Extract modules (file:class or file if no class)
    modules = []
    for entity in entities:
        # Extract module (class or just file if standalone function)
        if "." in entity.split(":")[-1]:
            # Has a class - extract it: "file.py:Class.method" → "file.py:Class"
            module = entity.rsplit(".", 1)[0]
        else:
            # No class - use full entity: "file.py:function" → "file.py:function"
            module = entity
        if module not in modules:
            modules.append(module)

    all_found_files = files
    all_found_modules = list(set(modules))
    all_found_entities = entities

    return all_found_files, all_found_modules, all_found_entities
