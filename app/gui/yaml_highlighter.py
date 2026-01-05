from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt6.QtCore import QRegularExpression

class YamlHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for YAML files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlighting_rules = []

        # Keys (e.g., "key:")
        key_format = QTextCharFormat()
        key_format.setForeground(QColor("#0074D9"))  # Blue
        key_format.setFontWeight(QFont.Weight.Bold)
        self._highlighting_rules.append(
            (QRegularExpression(r"^\s*[\w\-\._]+(?=:)"), key_format)
        )

        # Strings (double quotes)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#2ECC40"))  # Green
        self._highlighting_rules.append(
            (QRegularExpression(r"\".*\""), string_format)
        )
        
        # Strings (single quotes)
        self._highlighting_rules.append(
            (QRegularExpression(r"'.*'"), string_format)
        )

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B10DC9"))  # Markdown-ish purple
        self._highlighting_rules.append(
            (QRegularExpression(r"\b[0-9]+(\.[0-9]+)?\b"), number_format)
        )

        # Booleans
        bool_format = QTextCharFormat()
        bool_format.setForeground(QColor("#FF851B"))  # Orange
        bool_format.setFontWeight(QFont.Weight.Bold)
        self._highlighting_rules.append(
            (QRegularExpression(r"\b(true|false|True|False|yes|no)\b"), bool_format)
        )

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#AAAAAA"))  # Gray
        comment_format.setFontItalic(True)
        self._highlighting_rules.append(
            (QRegularExpression(r"#.*"), comment_format)
        )
        
        # List dashes
        dash_format = QTextCharFormat()
        dash_format.setForeground(QColor("#39CCCC"))  # Teal
        dash_format.setFontWeight(QFont.Weight.Bold)
        self._highlighting_rules.append(
            (QRegularExpression(r"^\s*-\s"), dash_format)
        )

    def highlightBlock(self, text):
        for pattern, format in self._highlighting_rules:
            expression = QRegularExpression(pattern)
            match = expression.match(text)
            while match.hasMatch():
                start = match.capturedStart()
                length = match.capturedLength()
                self.setFormat(start, length, format)
                match = expression.match(text, start + length)
