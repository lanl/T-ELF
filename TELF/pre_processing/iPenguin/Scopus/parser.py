import re
from typing import Union, List
from dataclasses import dataclass


class ScopusQueryParser:
    """
    ScopusQueryParser is a tool for parsing and normalizing boolean scopus queries with operators AND, OR, and AND NOT.

    This parser performs several tasks:
    1. Tokenization: Converts the input query string into a list of tokens.
    2. Syntax Validation: Checks for syntax errors such as unmatched parentheses and consecutive/orphaned operators.
    3. Parsing: Constructs an Abstract Syntax Tree (AST) based on the list of tokens.
    4. Normalization: Transforms the AST into a custom disjunctive form if possible. This form is loosely based on
       Disjunctive Normal Form (DNF). The current implementation does not support the normalization of complex 
       queries and can only handler queries with one level of nesting. A further effort needs to be undertaken to 
       provide such normalization for all possible Scopus queries.

    Constants:
    ----------
    PRECEDENCE: dict
        A dictionary to specify the precedence of the boolean operators.

    Data Classes:
    -------------
    ScopusValue: 
        A class representing leaf nodes in the AST. These nodes are the actual Scopus search filters e.x. 
        (AFFILCOUNTY(United States), AU-ID(1234567)). 
    ScopusOperation: 
        A class representing non-leaf nodes in the AST, which are the boolean operators, e.x. (AND, OR, AND NOT).

    Methods:
    --------
    tokenize(expression: str) -> List[str]:
        Converts the input boolean query string into a list of tokens.
    parse_expression(tokens: List[str]) -> Union[ScopusValue, ScopusOperation]:
        Constructs an Abstract Syntax Tree (AST) based on the list of tokens.
    build_ast(rpn_tokens: List[str]) -> Union[ScopusValue, ScopusOperation]:
        Builds the AST using the Reverse Polish Notation (RPN) of the tokens.
    normalize(expression: str) -> List[str]:
        Transforms the AST into a normalized form if possible and returns the separated expressions as strings.
    to_string(node: Union[ScopusValue, ScopusOperation]) -> str:
        Converts the AST to its string representation.
    """
    
    # set all allowed operators and their precedence
    # higher number signifies higher precedence
    PRECEDENCE = {'AND': 1, 'OR': 0, 'AND NOT': 1}

    
    def __init__(self):
        pass
    
    @dataclass
    class ScopusValue:
        """Node for values in the AST."""
        value: str
        
        def __len__(self):
            return 1

        
    @dataclass
    class ScopusOperation:
        """Node for operators in the AST."""
        operator: str
        operands: List[Union['ScopusValue', 'ScopusOperation']]

        def __post_init__(self):
            if self.operator not in ScopusQueryParser.PRECEDENCE.keys():
                raise ValueError(f'Invalid operator "{self.operator}". ' \
                                 f'Only {list(ScopusQueryParser.PRECEDENCE.keys())} are allowed.')

        def __len__(self):
            return self.total_parameters()

        def total_parameters(self) -> int:
            count = 0
            for op in self.operands:
                if isinstance(op, ScopusQueryParser.ScopusValue):
                    count += 1
                if isinstance(op, ScopusQueryParser.ScopusOperation):
                    count += op.total_parameters()
            return count
    
    
    def tokenize(self, expression: str) -> List[str]:
        """
        Tokenize the given boolean expression into a list of operators, parentheses, and values.

        Rules:
        1. Parameter names can be composed of alphabetical characters and hyphens.
        2. Parameter values are bounded by parentheses. A parameter's closing parenthesis is recognized 
           when it is followed by whitespace, an operator, or another closing parenthesis.
        3. Recognizes logical operators in self.PRECEDENCE.keys()

        Parameters:
        -----------
        expression: str
            The boolean expression to tokenize. The expression can include operators defined in the 
            PRECEDENCE constant, as well as parentheses and values that match the format 'Name(Argument)'.

        Returns:
        --------
        List[str]: 
            A list of tokens, where each token is either an operator, a parenthesis, or a value.

        Example:
        --------
        >>> expression = "(AU-ID(123) OR AU-ID(456)) AND LANGUAGE(english)"
        >>> parser = ScopusQueryParser()
        >>> parser. tokenize(expression)
        ['(', 'AU-ID(123)', 'OR', 'AU-ID(456)', ')', 'AND', 'LANGUAGE(english)']
        """
        i = 0
        tokens = []
        while i < len(expression):
            # 1. detect parameters
            filter_match = re.match(r'[A-Za-z\-]+\(', expression[i:])
            if filter_match:
                j = i + filter_match.end() - 1  # Starting from the detected '('
                paren_count = 1
                while j < len(expression) and paren_count != 0:
                    j += 1
                    if expression[j] == '(':
                        paren_count += 1
                    elif expression[j] == ')':
                        paren_count -= 1
                tokens.append(expression[i:j + 1])
                i = j + 1
                continue

            # 2. detect operators
            operator_match = re.match(r'AND NOT|AND|OR', expression[i:])
            if operator_match:
                tokens.append(operator_match.group(0))
                i += operator_match.end()
                continue

            # 3. tokenize standalone words
            word_match = re.match(r'[A-Za-z\-]+', expression[i:])
            if word_match:
                tokens.append(word_match.group(0))
                i += word_match.end()
                continue

            # 4. tokenize standalone parentheses
            if expression[i] in ['(', ')']:
                tokens.append(expression[i])
                i += 1
            else:
                i += 1
        return tokens

        
    def parse_expression(self, tokens: List[str]) -> Union[ScopusValue, ScopusOperation]:
        """
        Parse a list of tokens into an Abstract Syntax Tree (AST) representing the scopus query.

        This function uses the Shunting Yard algorithm to parse a list of tokens into an AST. The algorithm uses two stacks:
        1. `stack` holds values and completed sub-expressions.
        2. `op_stack` holds operators and unprocessed sub-expressions.

        The function also maintains a counter, `parenthesis_count`, to keep track of balanced parentheses in the expression. 
        A bad query with an odd number of parentheses will cause a ScopusQueryParser.SyntaxError. Consecutive or orphaned operators
        will also result in a ScopusQueryParser.SyntaxError.

        While iterating through the list of tokens, the function does the following:
          A. When an operator is encountered, it pops operators from `op_stack` to `stack` if they have higher or equal precedence, 
             then pushes the current operator onto `op_stack`.
          B. When an opening parenthesis is encountered, it increments `parenthesis_count` and pushes it onto `op_stack`.
          C. When a closing parenthesis is encountered, it decrements 'parenthesis_count' and pops all operators from '
             `op_stack` to `stack` until an opening parenthesis is encountered, which is then popped from `op_stack`.
          D. For all other tokens, it assumes they are Scopus filters (values) and pushes them onto `stack`.

        After parsing the input tokens, ScopusQueryParser.build_ast() is used to form the AST
            
            
        Parameters:
        -----------
        tokens: List[str]
            The list of tokens to parse into an AST. These tokens should be generated by ScopusQueryParser.tokenize()

        Returns:
        --------
        Union[ScopusValue, ScopusOperation]
            The root node of the generated AST. Returns None if the stack is empty.

        Raises:
        -------
        ScopusQueryParser.SyntaxError: 
            Raised for unbalanced parentheses or insufficient operands for an operator.
        """
        stack = []     # stack for holding values and completed sub-expressions
        op_stack = []  # stack for holding operators and unprocessed sub-expressions
        parenthesis_count = 0 
        last_token = None  # store last processed token to check for consecutive operators

        for token in tokens:
            if token in self.PRECEDENCE.keys():  # if token is an operator
                if last_token in list(self.PRECEDENCE.keys()) + ["("]:
                    raise self.SyntaxError("Consecutive or orphaned operators.")

                # pop operators from op_stack to stack if they have higher or equal precedence
                while op_stack and self.PRECEDENCE.get(op_stack[-1], -1) >= self.PRECEDENCE[token]:
                    stack.append(op_stack.pop())

                # push current operator onto op_stack
                op_stack.append(token)  

            elif token == "(":  # if token is opening parenthesis
                parenthesis_count += 1 
                op_stack.append(token)
            
            elif token == ")":  # if the token is closing parenthesis
                parenthesis_count -= 1 
                if parenthesis_count < 0:  # only < 0 when bad query
                    print(last_token)
                    raise self.SyntaxError("Unmatched closing parenthesis.")

                # move all operators from op_stack to stack until opening parenthesis is seen
                while op_stack and op_stack[-1] != "(":
                    stack.append(op_stack.pop())

                op_stack.pop()  # pop the opening parenthesis

            else:  # if token is a value
                stack.append(self.ScopusValue(token))

            # update last_token to the current token
            last_token = token  

        # query has been processed, parentheses counter should be 0
        if parenthesis_count != 0:
            raise self.SyntaxError("Unmatched opening parenthesis.")

        # clean up the op_stack
        while op_stack:
            stack.append(op_stack.pop())

        # reverse the stack for RPN and call the build function to get AST
        # see https://en.wikipedia.org/wiki/Reverse_Polish_notation for why reversal is done 
        return self.build_ast(stack[::-1])

    
    def build_ast(self, rpn_tokens):
        """
        Build the Abstract Syntax Tree (AST) from a list of tokens in Reverse Polish Notation (RPN).

        This function utilizes a stack-based approach to construct the AST from RPN tokens.
        Each token is either an operator or an operand (ScopusValue). For each operator, it pops 
        the two most recent operands from the stack and creates a new ScopusOperation.
        If the operands are already ScopusOperation objects with the same operator, they are combined 
        into one ScopusOperation to reduce redundancy.

        Parameters:
        -----------
        rpn_tokens: List[str]
            A list of tokens in Reverse Polish Notation, produced by ScopusQueryParser.parse_expression().

        Returns:
        --------
        ScopusValue, ScopusOperator, None
            The root node of the constructed AST, or None if the RPN token list is empty.

        Raises:
        -------
        SyntaxError: 
            If there are insufficient operands for an operator (bad query).
        """
        stack = []
        for token in reversed(rpn_tokens):
            
            if isinstance(token, self.ScopusValue):
                stack.append(token)
                
            else:  # if token is an operator
                if len(stack) < 2:
                    raise self.SyntaxError("Insufficient operands for operator.")

                # pop the right and left operands from the stack
                right_operand = stack.pop()
                left_operand = stack.pop()

                # if left operand is ScopusOperation with same operator, extend its operand list
                if isinstance(left_operand, self.ScopusOperation) and left_operand.operator == token:
                    left_operand.operands.append(right_operand)
                    stack.append(left_operand)

                # if right operand is ScopusOperation with same operator, extend its operand list
                elif isinstance(right_operand, self.ScopusOperation) and right_operand.operator == token:
                    right_operand.operands.insert(0, left_operand)
                    stack.append(right_operand)

                # create new ScopusOperation if neither can be combined
                else:
                    stack.append(self.ScopusOperation(token, [left_operand, right_operand]))

        # return root of AST
        return stack[0] if stack else None

    
    def parse(self, expression):
        """
        Take a Scopus query string and parse it to produce an AST that can be more easily
        manipulated to process the query

        Parameters:
        -----------
        expression: str
            The boolean expression to parse. The expression can include operators defined in the 
            PRECEDENCE constant, as well as parentheses and values that match the format 'Name(Argument)'.
        
        Returns:
        --------
        ScopusValue, ScopusOperator
            The root node of the constructed AST
        """
        
        tokens = self.tokenize(expression)  # tokenize the expression
        ast = self.parse_expression(tokens)  # parse the tokens into an AST
        return ast
        
        
    def split(self, expression):
        """
        Split an expression into two if possible

        This function takes a Scopus expression (as a string or query) and attempts to split the query
        into two logically equivalent queries if joined by OR. If such a split can be performed, the
        function returns a list of two elements where each element is one half of the split AST. If the
        split cannot be performed then the second element will be None and the first element will be
        unchanged.
        
        Parameters:
        -----------
        expression: str, ScopusValue, ScopusOperator
            If input is string, the expression will be first parsed to produce AST. Otherwise, AST will
            be used directly
            
        Returns:
        --------
        list: [c1, c2] -> c1 in (ScopusValue, ScopusOperator), c2 in (ScopusValue, ScopusOperator, None)
        """
        if not isinstance(expression, str):  # assume query has not been processed
            expression = self.to_string(expression)
        ast = self.parse(expression)
        normal_ast = self._normalize(ast)
        
        if isinstance(normal_ast, ScopusQueryParser.ScopusValue):  # if no operator
            return [normal_ast, None]
        elif len(normal_ast) < 2:  # if only one parameter
            return [normal_ast, None]
        elif normal_ast.operator != 'OR':  # if operator is not OR
            return [normal_ast, None]
        else:
            mid = len(normal_ast.operands) // 2
            chunk1 = normal_ast.operands[:mid]
            chunk2 = normal_ast.operands[mid:]
            return [
                ScopusQueryParser.ScopusOperation('OR', chunk1),
                ScopusQueryParser.ScopusOperation('OR', chunk2),
            ]
        
    
    def normalize(self, expression):
        """
        Normalize a Scopus boolean query by converting it to tokens, parsing it into an Abstract 
        Syntax Tree (AST), and then applying the __normalize function to optimize the AST. 
        This function aims to simplify the expression into globally OR-joined clauses if possible, 
        or otherwise return the original expression.

        Parameters:
        -----------
        expression: str, ScopusOperation, ScopusValue
            The input scopus query as a string.

        Returns:
        --------
        list
            A list containing the normalized expression(s) as strings. Each entry in the list can be 
            joined by an 'OR' and the value of the 'OR'.join(list) == expression.
            
        Example:
        --------
        >>> parser = ScopusQueryParser()
        >>> query1 = 'AU-ID(123) OR AU-ID(456) OR AU-ID(789)
        >>> parser.normalize(query1)
        ['AU-ID(123)', 'AU-ID(456)', 'AU-ID(789)']
        >>> 
        >>> query2 = '(AU-ID(123) OR AU-ID(456) OR AU-ID(789)) AND LANGUAGE(english)'
        >>> parser.normalize(query2)
        ['(AU-ID(123) AND LANGUAGE(english))',
         '(AU-ID(456) AND LANGUAGE(english))',
         '(AU-ID(789) AND LANGUAGE(english))']
        """
        if isinstance(expression, str):  # assume query has not been processed
            ast = self.parse(expression)
        else:
            ast = expression
            
        # normalize the AST
        normal_ast = self._normalize(ast)
        
        # convert back to string representation
        return [self.to_string(x) for x in self._separate_or_clauses(normal_ast)]

    
    def _normalize(self, node):
        """
        Normalize the AST node and its children recursively to optimize for globally OR-joined clauses. 
        
        This helper function attempts to simplify the expression without changing its semantics. If the 
        expression can be reduced then the simplification will be done. Otherwise, the function will return 
        the original expression as-is.
        
        Parameters:
        -----------
        ast: ScopusOperation, ScopusValue
            The AST root to normalize.

        Returns:
        --------
        ScopusOperation, ScopusValue
            The normalized AST root.
        """
        # normalize the AST node recursively
        if isinstance(node, ScopusQueryParser.ScopusValue):
            return node

        # normalize operands first
        normalized_operands = [self._normalize(op) for op in node.operands]

        # if this is an AND node and it has OR operands, distribute them
        if node.operator == 'AND' and any(isinstance(op, ScopusQueryParser.ScopusOperation) and op.operator == 'OR' for op in normalized_operands):
            return self._distribute_or(ScopusQueryParser.ScopusOperation(node.operator, normalized_operands))

        # if it's an OR node or an AND without OR operands, just recreate the node with normalized operands
        return ScopusQueryParser.ScopusOperation(node.operator, normalized_operands)

    
    def _distribute_or(self, and_node):
        """
        Distribute the OR operation across the AND operation in the AST, according to the 
        distributive laws of boolean algebra.

        This method takes an AND node and looks for any OR operands within. If it finds an OR 
        operand, it applies the distributive law to expand the expression into a form that adheres 
        to the Disjunctive Normal Form (DNF). Specifically, it transforms expressions of the form 
        (A OR B) AND C into (A AND C) OR (B AND C)

        Parameters
        ----------
        and_node: ScopusOperation
            The AND operation node which may contain OR operation operands that need to be distributed.

        Returns
        -------
        ScopusOperation
            A new Operation node that represents the AND operation with the OR operation distributed 
            across its operands.
        """
        # ensure that all operands of the AND operation are normalized
        normalized_operands = [self._normalize(op) for op in and_node.operands]

        # find first OR operand if any
        or_operand = next((op for op in normalized_operands if 
                           isinstance(op, ScopusQueryParser.ScopusOperation) and op.operator == 'OR'), None)

        # if no OR operand then there is nothing to distribute
        if or_operand is None:
            return and_node

        # remove the OR operand from the list
        normalized_operands.remove(or_operand)

        # distribute OR operand over the remaining AND operands
        distributed = []
        for term in or_operand.operands:
            
            # create new AND nodes by combining the term with all other operands of the original AND node
            new_and_operands = [term] + normalized_operands
            distributed.append(ScopusQueryParser.ScopusOperation('AND', new_and_operands))

        # return new OR operation combining all the distributed AND opertions
        return ScopusQueryParser.ScopusOperation('OR', distributed)


    def _separate_or_clauses(self, node):
        """
        Separate the clauses in an OR operation into a list of individual clause operations.

        This method recursively traverses an OR operation node in the AST and separates out each 
        clause (individual AND operations or values) into a list. It is assumed that the AST has 
        already been normalized and is in Disjunctive Normal Form (DNF), where each clause in an 
        OR operation should be a conjunction (AND operation) or a single value.

        Parameters
        ----------
        node: ScopusOperation
            The OR operation node to separate. It should only consist of AND operations or value nodes as operands.

        Returns
        -------
        List[ScopusOperation]
            A list of operations, where each operation is either a single ScopusValue or an AND operation representing
            an individual clause from the original OR operation.
        """
        if isinstance(node, ScopusQueryParser.ScopusValue):
            return [node]
        if node.operator == 'OR':
            or_clauses = []
            for op in node.operands:
                if isinstance(op, ScopusQueryParser.ScopusOperation) and op.operator == 'OR':
                    or_clauses.extend(self._separate_or_clauses(op))
                else:
                    or_clauses.append(op)
            return or_clauses
        else:
            # for non-OR operations, return a list with node itself
            return [node]
    
    
    @classmethod
    def to_string(cls, ast):
        """
        Convert an Abstract Syntax Tree (AST) to its string representation.

        Parameters:
        -----------
        ast: Union[ScopusValue, ScopusOperation]
            The AST node to be converted to string.

        Returns:
        --------
        str
            The string representation of the AST query.
        """
        if isinstance(ast, cls.ScopusValue):
            return ast.value

        # if the ast is an operation, process it
        if isinstance(ast, cls.ScopusOperation):
            operator = ast.operator  
            operands = ast.operands 

            if operator in ['AND', 'OR', 'AND NOT']:
                return "(" + f" {operator} ".join(cls.to_string(operand) for operand in operands) + ")"
            else:  # placeholder for unsupported operators (if any)
                return ""
        return ""

    
    class SyntaxError(Exception):
        """Custom exception for syntax errors."""
        pass