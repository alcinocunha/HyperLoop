"""
SMV Parser (by Claude Opus 4.5 adapted by Alcino Cunha)
Supports:  MODULE, VAR, FROZENVAR, INIT, TRANS, INVAR, and HLTLSPEC sections.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


# ============================================================================
# AST Node Definitions
# ============================================================================

class NodeType(Enum):
    # Literals
    INTEGER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    
    # Expressions
    UNARY_OP = auto()
    BINARY_OP = auto()
    NEXT_EXPR = auto()
    RANGE_EXPR = auto()
    QUANTIFIED_EXPR = auto()
    
    # Types
    BOOLEAN_TYPE = auto()
    RANGE_TYPE = auto()
    
    # Declarations
    VAR_DECL = auto()
    
    # Module
    MODULE = auto()


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    node_type: NodeType


@dataclass
class IntegerLiteral(ASTNode):
    value: int
    
    def __init__(self, value: int):
        super().__init__(NodeType.INTEGER)
        self.value = value
    
    def __repr__(self):
        return f"{self.value}"


@dataclass
class BooleanLiteral(ASTNode):
    value: bool
    
    def __init__(self, value: bool):
        super().__init__(NodeType.BOOLEAN)
        self.value = value
    
    def __repr__(self):
        return "TRUE" if self.value else "FALSE"


@dataclass
class Identifier(ASTNode):
    name: str
    trace: Optional[str] = None  # For trace quantification
    
    def __init__(self, name: str, trace: Optional[str] = None):
        super().__init__(NodeType.IDENTIFIER)
        self.name = name
        self.trace = trace
    
    def __repr__(self):
        if self.trace:
            return f"{self.name}[{self.trace}]"
        return self.name


@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode
    
    def __init__(self, operator: str, operand: ASTNode):
        super().__init__(NodeType.UNARY_OP)
        self.operator = operator
        self.operand = operand
    
    def __repr__(self):
        return f"({self.operator} {self.operand})"


@dataclass
class BinaryOp(ASTNode):
    operator: str
    left: ASTNode
    right: ASTNode
    
    def __init__(self, operator: str, left: ASTNode, right: ASTNode):
        super().__init__(NodeType.BINARY_OP)
        self.operator = operator
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class NextExpr(ASTNode):
    expr: ASTNode
    
    def __init__(self, expr:  ASTNode):
        super().__init__(NodeType.NEXT_EXPR)
        self.expr = expr
    
    def __repr__(self):
        return f"next({self.expr})"

# Type nodes
@dataclass
class BooleanType(ASTNode):
    def __init__(self):
        super().__init__(NodeType.BOOLEAN_TYPE)
    
    def __repr__(self):
        return "boolean"


@dataclass
class RangeType(ASTNode):
    lower: int
    upper: int
    
    def __init__(self, lower: int, upper: int):
        super().__init__(NodeType.RANGE_TYPE)
        self.lower = lower
        self.upper = upper
    
    def __repr__(self):
        return f"{self.lower}..{self. upper}"

@dataclass
class VarDecl(ASTNode):
    name: str
    var_type: ASTNode
    
    def __init__(self, name: str, var_type: ASTNode):
        super().__init__(NodeType.VAR_DECL)
        self.name = name
        self.var_type = var_type
    
    def __repr__(self):
        return f"{self.name} : {self. var_type}"


@dataclass
class QuantifiedExpr(ASTNode):
    quantifiers: list  # sequence of 'Forall' or 'Exists'
    vars: list
    expr: ASTNode
    
    def __init__(self, quantifiers: list, vars: list, expr: ASTNode):
        super().__init__(NodeType.QUANTIFIED_EXPR)
        self.quantifiers = quantifiers
        self.vars = vars
        self.expr = expr
    
    def __repr__(self):
        return ".".join([f"{q} {v}" for q, v in zip(self.quantifiers, self.vars)] + [f"({self.expr})"])

@dataclass
class Module(ASTNode):
    name: str
    var_decls: list = field(default_factory=list)
    frozenvar_decls: list = field(default_factory=list)
    init_expr: Optional[ASTNode] = None
    trans_expr: Optional[ASTNode] = None
    invar_expr: Optional[ASTNode] = None
    
    def __init__(self, name:  str):
        super().__init__(NodeType.MODULE)
        self.name = name
        self.var_decls = []
        self.frozenvar_decls = []
        self.init_expr = None
        self.trans_expr = None
        self.invar_expr = None
    
    def __repr__(self):
        parts = [f"MODULE {self.name}"]
        if self.frozenvar_decls:
            parts.append("FROZENVAR")
            for decl in self.frozenvar_decls:
                parts. append(f"    {decl};")
        if self.var_decls:
            parts. append("VAR")
            for decl in self.var_decls:
                parts.append(f"    {decl};")
        if self.init_expr:
            parts.append(f"INIT\n    {self.init_expr}")
        if self.trans_expr:
            parts.append(f"TRANS\n    {self. trans_expr}")
        if self.invar_expr:
            parts.append(f"INVAR\n    {self.invar_expr}")
        return "\n".join(parts)

# ============================================================================
# Tokenizer
# ============================================================================

class TokenType(Enum):
    # Keywords
    MODULE = auto()
    VAR = auto()
    FROZENVAR = auto()
    INIT = auto()
    TRANS = auto()
    INVAR = auto()
    NEXT = auto()
    TRUE = auto()
    FALSE = auto()
    BOOLEAN = auto()
    HLTLSPEC = auto()
    
    # Literals and identifiers
    INTEGER = auto()
    IDENTIFIER = auto()
    
    # Operators
    COLON = auto()       # :
    SEMICOLON = auto()   # ;
    COMMA = auto()       # ,
    DOT = auto()         # .
    DOTDOT = auto()      # ..
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    
    # Logical operators
    AND = auto()         # &
    OR = auto()          # |
    NOT = auto()         # !
    IMPLIES = auto()     # ->
    IFF = auto()         # <->
    
    # Comparison operators
    EQ = auto()          # =
    NEQ = auto()         # !=
    LT = auto()          # <
    LE = auto()          # <=
    GT = auto()          # >
    GE = auto()          # >=
    
    # Arithmetic operators
    PLUS = auto()        # +
    MINUS = auto()       # -
    TIMES = auto()       # *
    DIVIDE = auto()      # /
    MOD = auto()         # mod
    
    # Temporal operators
    ALWAYS = auto()      # G
    EVENTUALLY = auto()  # F
    AFTER = auto()       # X

    # Trace quantifiers
    FORALL = auto()      # Forall
    EXISTS = auto()      # Exists

    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    type: TokenType
    value:  any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


class Tokenizer:
    KEYWORDS = {
        'MODULE':  TokenType.MODULE,
        'VAR': TokenType.VAR,
        'FROZENVAR': TokenType.FROZENVAR,
        'INIT': TokenType.INIT,
        'TRANS': TokenType.TRANS,
        'INVAR':  TokenType.INVAR,
        'HLTLSPEC': TokenType.HLTLSPEC,
        'next': TokenType.NEXT,
        'TRUE': TokenType.TRUE,
        'FALSE': TokenType.FALSE,
        'boolean': TokenType.BOOLEAN,
        'mod': TokenType.MOD,
        'Forall': TokenType.FORALL,
        'Exists': TokenType.EXISTS,
        'G': TokenType.ALWAYS,
        'F': TokenType.EVENTUALLY,
        'X': TokenType.AFTER,
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
    
    def error(self, message: str):
        raise SyntaxError(f"Tokenizer error at line {self.line}, column {self.column}: {message}")
    
    def peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < len(self.text):
            return self.text[pos]
        return None
    
    def advance(self) -> Optional[str]:
        if self.pos < len(self. text):
            char = self.text[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None
    
    def skip_whitespace_and_comments(self):
        while self.pos < len(self.text):
            char = self.peek()
            
            # Skip whitespace
            if char in ' \t\r\n': 
                self.advance()
            # Skip single-line comments
            elif char == '-' and self.peek(1) == '-':
                while self.peek() and self.peek() != '\n':
                    self.advance()
            # Skip multi-line comments
            elif char == '/' and self.peek(1) == '*':
                self.advance()  # /
                self.advance()  # *
                while self.pos < len(self.text):
                    if self.peek() == '*' and self.peek(1) == '/':
                        self.advance()  # *
                        self. advance()  # /
                        break
                    self.advance()
            else:
                break
    
    def read_number(self) -> Token:
        start_line, start_col = self.line, self.column
        num_str = ''
        
        # Handle negative numbers
        if self. peek() == '-':
            num_str += self.advance()
        
        while self.peek() and self.peek().isdigit():
            num_str += self.advance()
        
        return Token(TokenType.INTEGER, int(num_str), start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line, start_col = self.line, self.column
        ident = ''
        
        while self.peek() and (self.peek().isalnum() or self.peek() in '_$#'):
            ident += self. advance()
        
        # Check if it's a keyword
        if ident in self.KEYWORDS:
            return Token(self. KEYWORDS[ident], ident, start_line, start_col)
        
        return Token(TokenType.IDENTIFIER, ident, start_line, start_col)
    
    def tokenize(self) -> list:
        self.tokens = []
        
        while self.pos < len(self.text):
            self.skip_whitespace_and_comments()
            
            if self.pos >= len(self.text):
                break
            
            char = self.peek()
            start_line, start_col = self.line, self.column
            
            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
            
            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            
            # Two-character operators
            elif char == ':' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, ':=', start_line, start_col))
            
            elif char == '.' and self.peek(1) == '.':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType. DOTDOT, '..', start_line, start_col))
            
            elif char == '-' and self.peek(1) == '>':
                self. advance()
                self.advance()
                self.tokens.append(Token(TokenType.IMPLIES, '->', start_line, start_col))
            
            elif char == '<' and self.peek(1) == '-' and self.peek(2) == '>':
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.IFF, '<->', start_line, start_col))
            
            elif char == '!' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NEQ, '!=', start_line, start_col))
            
            elif char == '<' and self.peek(1) == '=':
                self. advance()
                self.advance()
                self.tokens.append(Token(TokenType.LE, '<=', start_line, start_col))
            
            elif char == '>' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType. GE, '>=', start_line, start_col))
            
            # Single-character operators
            elif char == ':':
                self.advance()
                self.tokens.append(Token(TokenType.COLON, ':', start_line, start_col))
            
            elif char == ';':
                self.advance()
                self. tokens.append(Token(TokenType.SEMICOLON, ';', start_line, start_col))
            
            elif char == ',':
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ',', start_line, start_col))
            
            elif char == '.':
                self.advance()
                self.tokens.append(Token(TokenType.DOT, '.', start_line, start_col))
            
            elif char == '(':
                self. advance()
                self.tokens. append(Token(TokenType. LPAREN, '(', start_line, start_col))
            
            elif char == ')':
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', start_line, start_col))
            
            elif char == '{':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', start_line, start_col))
            
            elif char == '}':
                self.advance()
                self.tokens.append(Token(TokenType. RBRACE, '}', start_line, start_col))
            
            elif char == '[': 
                self.advance()
                self.tokens.append(Token(TokenType.LBRACKET, '[', start_line, start_col))
            
            elif char == ']':
                self.advance()
                self.tokens.append(Token(TokenType. RBRACKET, ']', start_line, start_col))
            
            elif char == '&':
                self.advance()
                self.tokens.append(Token(TokenType.AND, '&', start_line, start_col))
            
            elif char == '|':
                self.advance()
                self.tokens.append(Token(TokenType.OR, '|', start_line, start_col))
            
            elif char == '!':
                self.advance()
                self.tokens.append(Token(TokenType. NOT, '!', start_line, start_col))
            
            elif char == '=': 
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '=', start_line, start_col))
            
            elif char == '<':
                self.advance()
                self.tokens.append(Token(TokenType.LT, '<', start_line, start_col))
            
            elif char == '>':
                self.advance()
                self.tokens.append(Token(TokenType.GT, '>', start_line, start_col))
            
            elif char == '+':
                self.advance()
                self.tokens.append(Token(TokenType.PLUS, '+', start_line, start_col))
            
            elif char == '-':
                self. advance()
                self.tokens. append(Token(TokenType. MINUS, '-', start_line, start_col))
            
            elif char == '*':
                self.advance()
                self.tokens.append(Token(TokenType. TIMES, '*', start_line, start_col))
            
            elif char == '/':
                self.advance()
                self. tokens.append(Token(TokenType.DIVIDE, '/', start_line, start_col))
            
            else:
                self.error(f"Unexpected character:  {char !r}")
        
        self.tokens.append(Token(TokenType.EOF, None, self. line, self.column))
        return self.tokens


# ============================================================================
# Parser
# ============================================================================

class Parser: 
    # Section keywords that can start a new section
    SECTION_KEYWORDS = {
        TokenType.VAR, TokenType.FROZENVAR, TokenType.INIT,
        TokenType.TRANS, TokenType.INVAR, TokenType.HLTLSPEC, TokenType.MODULE
    }
    
    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0
    
    def error(self, message: str):
        token = self.current()
        raise SyntaxError(f"Parse error at line {token.line}, column {token.column}: {message}")
    
    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF
    
    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def advance(self) -> Token:
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self. pos += 1
        return token
    
    def expect(self, token_type: TokenType, message: str = None) -> Token:
        if self.current().type != token_type:
            msg = message or f"Expected {token_type}, got {self.current().type}"
            self.error(msg)
        return self. advance()
    
    def match(self, *token_types:  TokenType) -> bool:
        return self.current().type in token_types
    
    def is_section_keyword(self) -> bool:
        return self.current().type in self.SECTION_KEYWORDS
    
    # ========================================================================
    # Expression Parsing (Precedence Climbing)
    # ========================================================================
    
    def parse_expression(self) -> ASTNode:
        return self.parse_iff()
    
    def parse_iff(self) -> ASTNode:
        left = self.parse_implies()
        while self.match(TokenType.IFF):
            op = self.advance().value
            right = self.parse_implies()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_implies(self) -> ASTNode:
        left = self.parse_or()
        while self.match(TokenType.IMPLIES):
            op = self.advance().value
            right = self.parse_or()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match(TokenType. OR):
            op = self. advance().value
            right = self.parse_and()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_and(self) -> ASTNode:
        left = self.parse_comparison()
        while self.match(TokenType.AND):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_additive()
        while self.match(TokenType.EQ, TokenType.NEQ, TokenType. LT, 
                         TokenType.LE, TokenType.GT, TokenType.GE):
            op = self.advance().value
            right = self.parse_additive()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType. MINUS):
            op = self. advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        left = self. parse_unary()
        while self.match(TokenType. TIMES, TokenType.DIVIDE, TokenType.MOD):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.NOT):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.MINUS):
            op = self. advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.ALWAYS):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.EVENTUALLY):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.AFTER):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        token = self.current()
        
        # Integer literal
        if self.match(TokenType.INTEGER):
            self.advance()
            return IntegerLiteral(token.value)
        
        # Boolean literals
        if self.match(TokenType.TRUE):
            self.advance()
            return BooleanLiteral(True)
        
        if self.match(TokenType.FALSE):
            self.advance()
            return BooleanLiteral(False)
        
        # next(expr)
        if self.match(TokenType.NEXT):
            self.advance()
            self.expect(TokenType.LPAREN)
            expr = self.parse_identifier()
            self.expect(TokenType.RPAREN)
            return NextExpr(expr)
                
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        # Identifier
        if self.match(TokenType.IDENTIFIER):
            return self.parse_identifier()
        
        self.error(f"Unexpected token in expression: {token}")
    
    def parse_identifier(self) -> ASTNode:
        name = self.expect(TokenType.IDENTIFIER).value
        trace = None
        # Check for trace quantification
        if self.match(TokenType.LBRACKET):
            self.advance()  # [
            trace = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.RBRACKET)  # ]
            result = Identifier(name, trace)
        else:
            result = Identifier(name)        
        return result
    
    def parse_quantified_expression(self) -> ASTNode:
        quantifiers = []
        vars = []
        
        while self.match(TokenType.FORALL, TokenType.EXISTS):
            quant_token = self.advance()
            quantifiers.append(quant_token.value)
            var_name = self.expect(TokenType.IDENTIFIER).value
            vars.append(var_name)
            self.expect(TokenType.DOT)
        
        expr = self.parse_expression()
        return QuantifiedExpr(quantifiers, vars, expr)
    
    # ========================================================================
    # Type Parsing
    # ========================================================================
    
    def parse_type(self) -> ASTNode:
        # boolean type
        if self.match(TokenType. BOOLEAN):
            self.advance()
            return BooleanType()
        
        # Range type:  lower..upper
        if self.match(TokenType.INTEGER):
            lower = self.advance().value
            self.expect(TokenType. DOTDOT)
            upper = self.expect(TokenType.INTEGER).value
            return RangeType(lower, upper)
        
        self.error(f"Expected type, got {self.current()}")
    
    # ========================================================================
    # Section Parsing
    # ========================================================================
    
    def parse_var_section(self, is_frozen: bool = False) -> list:
        """Parse VAR or FROZENVAR section."""
        if is_frozen:
            self.expect(TokenType.FROZENVAR)
        else:
            self.expect(TokenType.VAR)
        
        decls = []
        
        while self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            self.expect(TokenType.COLON)
            var_type = self.parse_type()
            self.expect(TokenType.SEMICOLON)
            decls.append(VarDecl(name, var_type))
        
        return decls
    
    def parse_init_section(self) -> ASTNode:
        """Parse INIT section."""
        self.expect(TokenType.INIT)
        expr = self.parse_expression()
        return expr
    
    def parse_trans_section(self) -> ASTNode:
        """Parse TRANS section."""
        self.expect(TokenType.TRANS)
        expr = self.parse_expression()
        return expr
    
    def parse_invar_section(self) -> ASTNode:
        """Parse INVAR section."""
        self.expect(TokenType.INVAR)
        expr = self.parse_expression()
        return expr
    
    # ========================================================================
    # Module Parsing
    # ========================================================================
    
    def parse_module(self) -> Module:
        """Parse a MODULE declaration."""
        self.expect(TokenType.MODULE)
        
        name = self.expect(TokenType. IDENTIFIER).value
        module = Module(name)
        
        # Parse sections
        while not self.match(TokenType.EOF, TokenType.MODULE):
            if self.match(TokenType.VAR):
                module.var_decls.extend(self.parse_var_section(is_frozen=False))
            elif self.match(TokenType.FROZENVAR):
                module.frozenvar_decls.extend(self.parse_var_section(is_frozen=True))
            elif self.match(TokenType. INIT):
                module.init_expr = self.parse_init_section()
            elif self.match(TokenType.TRANS):
                module.trans_expr = self. parse_trans_section()
            elif self.match(TokenType. INVAR):
                module. invar_expr = self.parse_invar_section()
            else:
                self.error(f"Unexpected token in module:  {self.current()}")
        
        return module

# ============================================================================
# Main Parser Function
# ============================================================================

def parse_smv(text: str) -> Module:
    """
    Parse SMV code and return a Module AST nodes.
    
    Args:
        text: The SMV source code as a string. 
    
    Returns:
        A Module object representing the parsed module.
    
    Raises:
        SyntaxError: If the input contains invalid syntax.
    """
    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse_module()

def parse_hyperltl(text: str) -> ASTNode:
    """
    Parse a HyperLTL expression and return the corresponding AST node.
    
    Args:
        text: The HyperLTL expression as a string.
    
    Returns:
        An ASTNode representing the parsed HyperLTL expression.

    Raises:
        SyntaxError: If the input contains invalid syntax.
    """
    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse_quantified_expression()
