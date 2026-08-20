"""Tree-sitter-based comment removal that preserves source coordinates."""

from collections import Counter


COMMENT_NODE_TYPES = frozenset({
    'comment',
    'line_comment',
    'block_comment',
    'jsx_comment',
    'html_comment',
})


def comment_nodes(root_node):
    """Return outermost tree-sitter comment nodes in source order."""
    found = []

    def visit(node):
        if node.type in COMMENT_NODE_TYPES:
            found.append(node)
            return
        for child in node.children:
            visit(child)

    visit(root_node)
    return sorted(found, key=lambda node: (node.start_byte, node.end_byte))


def strip_comments(code, parser):
    """Replace actual comment bytes with spaces while preserving newlines.

    Using parser nodes instead of a regular expression protects comment-like
    text inside strings, Java text blocks, Go raw strings, JavaScript template
    literals, and JavaScript regular-expression literals. Replacing bytes (as
    opposed to characters) preserves every tree-sitter byte coordinate.
    """
    source = code.encode('utf-8')
    tree = parser.parse(source)
    if tree.root_node.has_error:
        raise ValueError('Tree-sitter reported a parse error before preprocessing')

    nodes = comment_nodes(tree.root_node)
    cleaned = bytearray(source)
    for node in nodes:
        for index in range(node.start_byte, node.end_byte):
            if cleaned[index] not in {10, 13}:  # Preserve LF and CR.
                cleaned[index] = 32

    cleaned_code = cleaned.decode('utf-8')
    cleaned_tree = parser.parse(bytes(cleaned))
    if cleaned_tree.root_node.has_error:
        raise ValueError('Tree-sitter reported a parse error after preprocessing')
    remaining = comment_nodes(cleaned_tree.root_node)
    if remaining:
        raise ValueError('Comment nodes remain after preprocessing')

    return cleaned_code, {
        'comments_removed': len(nodes),
        'comment_bytes_removed': sum(
            node.end_byte - node.start_byte for node in nodes
        ),
        'comment_types': dict(Counter(node.type for node in nodes)),
    }
