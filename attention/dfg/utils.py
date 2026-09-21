import re
from io import StringIO
import tokenize

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        # This else-block naturally handles C-style comments (// and /* */)
        # used by Java, Go, and JavaScript!
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " 
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

def tree_to_token_index(
    root_node,
    include_comments=False,
    atomic_string_literals=False,
):
    ignore_types = [
        'comment', 'line_comment', 'block_comment',
        'jsx_comment', 'html_comment', 'program', 'statement_block', 'string_fragment'
    ]
    if include_comments:
        ignore_types = [
            node_type for node_type in ignore_types
            if node_type not in {
                'comment', 'line_comment', 'block_comment',
                'jsx_comment', 'html_comment',
            }
        ]

    is_atomic_string = (
        atomic_string_literals and root_node.type == 'string_literal'
    )
    if (
        len(root_node.children) == 0
        or root_node.type == 'string'
        or is_atomic_string
    ) and root_node.type not in ignore_types:
        if root_node.start_point == root_node.end_point:
            return []
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(
                child,
                include_comments=include_comments,
                atomic_string_literals=atomic_string_literals,
            )
        return code_tokens

def tree_to_variable_index(root_node, index_to_code):
    ignore_types = [
        'comment', 'line_comment', 'block_comment',
        'jsx_comment', 'html_comment',
        'program', 'statement_block', 'string_fragment'
    ]

    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type not in ignore_types:
        if root_node.start_point == root_node.end_point:
            return []
        index = (root_node.start_point, root_node.end_point)

        # --- THE SAFETY GUARD TO PREVENT KEYERROR ---
        if index not in index_to_code:
            return []

        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens

def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s
