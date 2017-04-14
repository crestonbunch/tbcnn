NODE_LIST = [
    'Module','Interactive','Expression','FunctionDef','ClassDef','Return',
    'Delete','Assign','AugAssign','Print','For','While','If','With','Raise',
    'TryExcept','TryFinally','Assert','Import','ImportFrom','Exec','Global',
    'Expr','Pass','Break','Continue','attributes','BoolOp','BinOp','UnaryOp',
    'Lambda','IfExp','Dict','Set','ListComp','SetComp','DictComp',
    'GeneratorExp','Yield','Compare','Call','Repr','Num','Str','Attribute',
    'Subscript','Name','List','Tuple','Load','Store','Del',
    'AugLoad','AugStore','Param','Ellipsis','Slice','ExtSlice','Index','And','Or',
    'Add','Sub','Mult','Div','Mod','Pow','LShift','RShift','BitOr','BitXor',
    'BitAnd','FloorDiv','Invert','Not','UAdd','USub','Eq','NotEq','Lt',
    'LtE','Gt','GtE','Is','IsNot','In','NotIn','comprehension','ExceptHandler',
    'arguments','keyword','alias']

NODE_MAP = {x: i for (i, x) in enumerate(NODE_LIST)}
