import os
import sys
import unittest


ATTENTION_DIR = os.path.dirname(os.path.abspath(__file__))
if ATTENTION_DIR not in sys.path:
    sys.path.insert(0, ATTENTION_DIR)

from comment_preprocessing import COMMENT_NODE_TYPES, comment_nodes, strip_comments
from dfg_comp import build_parser
from graph_utils import traverse_node
from preprocess_csn_comments import aligned_token_info, preprocess_record


class CommentPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsers = {
            language: build_parser(language)
            for language in ('java', 'go', 'javascript')
        }

    def parser_tokens(self, language, code):
        byte_code = code.encode('utf-8')
        tree = self.parsers[language].parse(byte_code)
        self.assertFalse(tree.root_node.has_error)
        collected = []
        traverse_node(
            tree.root_node,
            collected,
            byte_code,
            include_comments=True,
        )
        return [info['token'] for info in collected]

    def check_language(self, language, code, preserved_fragments):
        parser = self.parsers[language]
        tokens = self.parser_tokens(language, code)
        record = {'code': code, 'code_tokens': tokens, 'partition': 'test'}
        cleaned, stats = preprocess_record(record, parser, language)

        self.assertGreater(stats['comments_removed'], 0)
        self.assertEqual(
            stats['comments_removed'], stats['comment_tokens_removed']
        )
        self.assertEqual(
            len(code.encode('utf-8')), len(cleaned['code'].encode('utf-8'))
        )
        self.assertEqual(code.count('\n'), cleaned['code'].count('\n'))
        for fragment in preserved_fragments:
            self.assertIn(fragment, cleaned['code'])
        tree = parser.parse(cleaned['code'].encode('utf-8'))
        self.assertFalse(tree.root_node.has_error)
        self.assertFalse(comment_nodes(tree.root_node))
        self.assertFalse(
            any(
                info['type'] in COMMENT_NODE_TYPES
                for info in aligned_token_info(
                    cleaned['code'],
                    cleaned['code_tokens'],
                    parser,
                    include_comments=False,
                )
            )
        )
        cleaned_twice, second_stats = preprocess_record(
            cleaned, parser, language
        )
        self.assertEqual(cleaned_twice, cleaned)
        self.assertEqual(second_stats['comments_removed'], 0)
        self.assertEqual(second_stats['comment_tokens_removed'], 0)

    def test_java_comments_are_removed_but_strings_are_preserved(self):
        self.check_language(
            'java',
            'void f(){ String s="// keep"; /** remove */ int x=1; // gone\n}',
            ['"// keep"'],
        )

    def test_go_comments_are_removed_but_raw_strings_are_preserved(self):
        self.check_language(
            'go',
            'func f(){ s := `/* keep */`; /* remove */ _ = s // gone\n}',
            ['`/* keep */`'],
        )

    def test_javascript_comments_are_removed_but_templates_are_preserved(self):
        self.check_language(
            'javascript',
            'function f(){ const s=`// keep`; /** remove */ return s; // gone\n}',
            ['`// keep`'],
        )

    def test_strip_comments_is_a_noop_on_comment_free_code(self):
        code = 'function f(){ return "// text"; }'
        cleaned, stats = strip_comments(code, self.parsers['javascript'])
        self.assertEqual(cleaned, code)
        self.assertEqual(stats['comments_removed'], 0)

    def test_known_comment_node_types_are_never_retained(self):
        self.assertIn('comment', COMMENT_NODE_TYPES)
        self.assertIn('line_comment', COMMENT_NODE_TYPES)
        self.assertIn('block_comment', COMMENT_NODE_TYPES)


if __name__ == '__main__':
    unittest.main()
