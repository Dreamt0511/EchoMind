from langchain_core.runnables import RunnableLambda
#func = lambda ai_msg : {"name":ai_message.content}
#方法一，使用my_func = RunnableLambda(func)转换函数后，再把func加入链
#chain = first_prompt | model | my_func |second_prompt | model | str_parser

#方法二，直接把函数加入链，类似下面这样
#chain = first_prompt | model | (lambda ai_msg : {"name":ai_message.content}) |second_prompt | model | str_parser
