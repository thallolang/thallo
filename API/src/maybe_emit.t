return {
  name = "maybe_emit"; --name for debugging
  -- list of keywords that will start our expressions
  entrypoints = {"maybe_emit"};
  keywords = {}; --list of keywords specific to this language
   --called by Terra parser to enter this language
  expression = function(self,lex)
    lex:expect("maybe_emit")
    lex:expect("(")
    local pred = lex:luaexpr()
    lex:expect(",")
    local qt = lex:luaexpr()
    lex:expect(")")
    return function(environment_function)
        local env = environment_function()
        if pred(env) then
            local snippet = qt(env)
            return snippet
        end
        return quote end
    end
  end;
}