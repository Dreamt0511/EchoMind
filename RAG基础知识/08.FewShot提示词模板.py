from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_community.llms.tongyi import Tongyi

#示例数据的模板
example_template = PromptTemplate.from_template("单词{word}，反义词{antonym}")

#示例数据
examples_data = [
    {"word":"big","antonym":"small"},
    {"word":"left","antonym":"right"}
]

few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,#示例数据的模板
    examples = examples_data,#示例数据
    prefix = "告知我单词的反义词，我提供的示例如下",#示例之前的提示词
    suffix = "回复格式严格基于前面的示例告诉我，{input_word}的反义词是？",#示例之后的提示词
    input_variables = ["input_word"]#声明在前缀和后缀中需要注入的变量名
)
#旧写法
# prompt_text = few_shot_template.format_prompt(input_word = "up").to_string()
#新版写法
prompt_text = few_shot_template.invoke(input={"input_word":"up"}).to_string()
print(prompt_text)

model = Tongyi(model="qwen-plus")
print(model.invoke(input=prompt_text))
print(type(model.invoke(input=prompt_text)))


