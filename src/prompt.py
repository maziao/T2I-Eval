EXTRACT_TEMPLATE = """# Your task
You are an expert in information extraction. Your task is to extract attributes of entities and relationships between entities from the text, and to pose questions about each entity's attributes and relationships. You can also generate proper image caption based on the given image.

# Input data
The text is: {text_prompt}
The image is: <ImagePlaceholder>. The image can only be used to generate image caption, and must be ignored when extracting entities and generating questions.

# Extract pipeline
Step 1: Identify Entities
    Step 1.1: Extract All Nouns
        Extract all potential nouns from the input text.
    Step 1.2: Evaluate Each Noun
        Determine Entity Status: For each extracted noun, evaluate whether it is an entity based on context and predefined criteria.
        Include or Exclude: If the noun is determined to be an entity, include it in the output. If the noun is not determined to be an entity, exclude it from the output.
Step 2: For each entity, create a critical question regarding whether the appearance of the entity in the generated image is realistic, aesthetically pleasing, and aligns with human intuition. Questions should focus primarily on overall authenticity and not get too detailed.
Step 3: Identify All Attributes for Each Entity
    Step 3.1: Identify Intrinsic Attributes
        Intrinsic attributes are properties of the entity explicitly mentioned in the input text, such as color, size, shape, material, quantity, etc.
        Step 3.1.1: Extract Quantity Attributes
            First, analyze the input text specifically for words representing quantity, including words like "a" or "an" which indicate a quantity of one. For example, in the phrase "a cat", "a" indicates a quantity of one cat. Identify the entity they refer to and set the quantity attribute of that entity to this value.
        Step 3.1.2: Extract Other Intrinsic Attributes
            For each entity, analyze the remaining words in the input text related to it (excluding the entity name itself).
            Determine if a word is an intrinsic attribute of the entity. Identify the attribute type (e.g., color, size, material) and value. The attribute value is derived from the text.
        Step 3.1.3: Verify the attribute type and value pair. If the attribute value does not appear in the input text, appears in the entity name, or is "unspecified", ignore this pair.
        Step 3.1.4: Ignore attribute type and value pairs if the attribute type is related to position, orientation, distance, location, etc.
        Step 3.1.5: For each entity, add an additional attribute: existence with a value of "yes" to indicate that the entity should exist in the image.
        Step 3.1.6: If the input text does not specify a quantity, set the quantity attribute to “unspecified”.
        Step 3.1.7: Add the verified attribute type and value pairs to the output. All the entities should be in the output.
    Step 3.2: Identify Relational Attributes
        Relational attributes describe the entity's relationship with other entities.
        Step 3.2.1: Analyze all words related to entities in the input text. If a word is a relational attribute between entities, identify the relationship type and the related entities.
        Step 3.2.2: Add the relationship types and related entities to the output.
Step 4: Construct Questions Based on the Extracted Attributes
    Step 4.1: Construct Intrinsic Attribute Consistency Questions
        Step 4.1.1: For the existence attribute, generate a question like: “Does the [entity] exist in the image?” where [entity] is the name of the entity.
        Step 4.1.2: For each entity's intrinsic attribute pair (type and value), generate a question asking about the attribute value of the entity. Ensure that every intrinsic attribute type-value pair results in one question. 
        Step 4.1.3: Ensure that the total number of questions matches the total number of other intrinsic attribute-value pairs plus the existence question and quantity question for each entity. Each intrinsic attribute type-value pair should correspond to one question.
    Step 4.2: Construct Relationship Attribute Consistency Questions
        Step 4.2.1: For each relationship attribute of each entity, generate a question asking about the relationship value between the entity and other entities.
        Step 4.2.2: Ensure that the number of questions generated matches the number of relationship attribute pairs. Each relationship pair should correspond to one question.
Step 5: Generate image caption
    Step 5.1: Identify all entities in the image.
    Step 5.2: For each entity, generate a caption that includes the entity's name and all attributes of the entity.
    Step 5.3: For each entity, generate a caption that includes the entity's name and all relationships of the entity.

# Output template
Repalce Variable in `{{}}`
And if the text is like "Three apples", the entity should be "apple", and the attribute should be "three". Instead of "apple 1, apple2, apple3" as the entities.
When generating attribute consistency questions, the answer to the question should be strongly correlated with the given attribute in the type. For example, for the attribute 'old', the answer for the question construed with the generated attribute type shoud be 'old' or 'new'.
Please generate your extracted structure information based on following markdown template (Do NOT generate // comment in the template)
# Structure Information
## Intrinsic Attributes
### {{Entity}}
- attribute 1: {{attribute 1 type}}: {{attribute 1 value}}
- attribute 2: {{attribute 2 type}}: {{attribute 2 value}}
- attribute 3: {{attribute 3 type}}: {{attribute 3 value}}
...
### {{Next Entity or Group}}
...
## Relationship Attributes
### {{Relationship 1 Name}}
- entities involved: {{entity1, entity2, ...}}
- value: {{relationship value}}
### {{Next Relationship}}
...

# Questions
## Appearance Quality Questions
### {{entity 1 name}}
- question1: {{entity 1 Appearance Quality Question 1}}
### {{next entity}}
...

## Intrinsic Attribute Consistency Questions
### {{entity 1 name}}
- question1: {{entity 1 Intrinsic Attribute Consistency Question 1}}
- question2: {{entity 1 Intrinsic Attribute Consistency Question 2}}
- next question
...
### {{next entity}}
...

## Relationship Attribute Consistency Questions
- question1: {{Relationship Attribute Consistency Question 1}}
    - entities: {{entity1}} {{entity2}}
- question2: {{Relationship Attribute Consistency Question2}}
...

# Image Caption
## {{Entity 1 name}}
- caption: {{entity1 caption}}
## {{Next Entity}}
...
"""


REF_BASED_ANSWER_TEMPLATE_ABLATION_1 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The questions you should answer. The questions are focused on three aspects: appearance quality, intrinsic attribute consistency, and relationship attribute consistency of entities in the image. The questions are:
{questions}
2. The target image: <ImagePlaceholder>; You need to answer the questions based on the content of this image.
3. The reference image: <ImagePlaceholder>.

# Guidelines
* Step1: Answer the appearance quality questions
    Answer each question separately. For each question,
    - Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
    - Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
    - You can use the reference image as a reference for authenticity when you answer questions about appearance quality based on the target image.
    - Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
    - Give a brief explanation for the score you give.
    - The reference image is only used for answering appearance quality questions.
* Step2: Answer the intrinsic attribute consistency questions
    Answer each question separately. For each question,
    - Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
    - Write out the attribute value of the entity in the target image. Your anwser should be based on the target image and the target image caption.
* Step3: Answer the relationship attribute consistency questions
    Answer each question separately. For each question,
    - Determine whether the entities from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
    - Write out the relationships of the entities in the target image. Your anwser should be based on the target image and the target image caption.

# Scoring strategy (for appearance quality questions only)
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## Appearance Quality Answers
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
    - score: {{score}}
...
## Intrinsic Attribute Consistency Answers
...
## Relationship Attribute Consistency Answers
...
"""


REF_FREE_ANSWER_TEMPLATE_ABLATION_1 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The questions you should answer. The questions are focused on three aspects: appearance quality, intrinsic attribute consistency, and relationship attribute consistency of entities in the image. The questions are:
{questions}
2. The target image: <ImagePlaceholder>; You need to answer the questions based on the content of this image.

# Guidelines
* Step1: Answer the appearance quality questions
    Answer each question separately. For each question,
    - Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
    - Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
    - Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
    - Give a brief explanation for the score you give.
* Step2: Answer the intrinsic attribute consistency questions
    Answer each question separately. For each question,
    - Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
    - Write out the attribute value of the entity in the target image. Your anwser should be based on the target image and the target image caption.
* Step3: Answer the relationship attribute consistency questions
    Answer each question separately. For each question,
    - Determine whether the entities from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
    - Write out the relationships of the entities in the target image. Your anwser should be based on the target image and the target image caption.

# Scoring strategy (for appearance quality questions only)
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## Appearance Quality Answers
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
    - score: {{score}}
...
## Intrinsic Attribute Consistency Answers
...
## Relationship Attribute Consistency Answers
...
"""


EVAL_TEMPLATE_ABLATION_1 = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate. The (question, answer) pair may be focused on three aspects: appearance quality, intrinsic attribute consistency, and relationship attribute consistency of entities in the image. The answers are:
{answers}

# Guidelines
* Step1: Evaluate the appearance quality answers
    - Do nothing, just follow the answers' appearance quality.
* Step2 & 3: Evaluate the intrinsic & relationship attribute consistency answers
    Evaluate each answer separately. For each answer,
    - Compare the answer with the structure information.
    - If the answer is that the entity does not exist in the image, give a score of 0 for the answer. If not, proceed to the next step.
    - Give a brief description of how well the answer matches the ground truth.
    - Give a score from 0 to 10 for the answer according to the scoring strategy and the description above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
## Appearance Quality Answers
### {{entity name}}
- question: {{question}}
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
    - score: {{score}}
...
## Intrinsic Attribute Consistency Answers
...
## Relationship Attribute Consistency Answers
...
"""


REF_BASED_APPEARANCE_ANSWER_TEMPLATE = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The question you should answer. The question is focused on the appearance quality of entities in the image. The question is:
{question}
2. The target image: <ImagePlaceholder>; You need to answer the question based on the content of this image.
3. The reference image: <ImagePlaceholder>.

# Guidelines
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
- Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
- You can use the reference image as a reference for authenticity when you answer questions about appearance quality based on the target image.
- Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
- Give a brief explanation for the score you give. 

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
    - score: {{score}}
"""


REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_1 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The question you should answer. The question is focused on appearance quality of entities in the image. The question is:
{question}
2. The target image: <ImagePlaceholder>; You need to answer the question based on the content of this image.
3. The reference image: <ImagePlaceholder>.

# Guidelines
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
- Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
- You can use the reference image as a reference for authenticity when you answer questions about appearance quality based on the target image.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
"""


REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_2 = """# Your task
You are an assistant specialized in scoring the appearance quality of images.

# Input data
1. Question and explanation. The question is focused on the appearance quality of entities in the image. The question and explanation is:
{question_and_exp}
2. The target image: <ImagePlaceholder>; You need to score the appearance quality of this image.
3. The reference image: <ImagePlaceholder>.

# Guidelines
- Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
- You can use the reference image as a reference for authenticity when you give a score based on the target image.

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


REF_BASED_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The questions you should answer. The questions are focused on the appearance quality of entities in the image. The questions are:
{questions}
2. The target image: <ImagePlaceholder>; You need to answer the questions based on the content of this image.
3. The reference image: <ImagePlaceholder>.

# Guidelines
Answer each question separately. For each question,
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
- Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
- You can use the reference image as a reference for authenticity when you answer questions about appearance quality based on the target image.
- Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
- Give a brief explanation for the score you give. 

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
    - score: {{score}}
...
"""


REF_FREE_APPEARANCE_ANSWER_TEMPLATE = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The question you should answer. The question is focused on appearance quality of entities in the image. The question is:
{question}
2. The target image: <ImagePlaceholder>. You need to answer the question based on the content of this image.

# Guidelines
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
- Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
- Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
- Give a brief explanation for the score you give. 

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
    - score: {{score}}
"""


REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_1 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The question you should answer. The question is focused on appearance quality of entities in the image. The question is:
{question}
2. The target image: <ImagePlaceholder>. You need to answer the question based on the content of this image.

# Guidelines
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
- Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
"""


REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_2 = """# Your task
You are an assistant specialized in scoring the appearance quality of images.

# Input data
1. Question and explanation. The question is focused on appearance quality of entities in the image. The question and explanation is:
{question_and_exp}
2. The target image: <ImagePlaceholder>. You need to grade the appearance quality of this image.

# Guidelines
- Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


REF_FREE_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The questions you should answer. The questions are focused on appearance quality of entities in the image. The questions are:
{questions}
2. The target image: <ImagePlaceholder>. You need to answer the questions based on the content of this image.

# Guidelines
Answer each question separately. For each question,
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, give a score of 0 for the question.
- Answer whether the appearance of the entity in the target image is realistic, aesthetically pleasing, and aligns with human intuition.
- Give a score from 0 to 10 for each question. 0 means the appearance is not realistic, aesthetically pleasing, or align with human intuition at all, and 10 means the appearance is very realistic, aesthetically pleasing, or align with human intuition.
- Give a brief explanation for the score you give. 

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - explanation: {{explanation}}
    - score: {{score}}
...
"""


INTRINSIC_ANSWER_TEMPLATE = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The question you should answer. The question is focused on intrinsic attribute consistency of entities in the image. The question is:
{question}
2. The target image: <ImagePlaceholder>. You need to answer the question based on the content of this image.

# Guidelines
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
- Write out the attribute value of the entity in the target image. Your anwser should be based on the target image and the target image caption.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - answer: {{answer}}
"""


INTRINSIC_ANSWER_TEMPLATE_ABLATION_2 = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The questions you should answer. The questions are focused on intrinsic attribute consistency of entities in the image. The questions are:
{questions}
2. The target image: <ImagePlaceholder>. You need to answer the questions based on the content of this image.

# Guidelines
Answer each question separately. For each question,
- Determine whether the entity from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
- Write out the attribute value of the entity in the target image. Your anwser should be based on the target image and the target image caption.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
## {{entity name}}
- question: {{question}}
    - answer: {{answer}}
...
"""


RELATIONSHIP_ANSWER_TEMPLATE  = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The question you should answer. The question is focused on relationship attribute consistency of entities in the image. The question is:
{question}
2. The target image: <ImagePlaceholder>. You need to answer the question based on the content of this image.

# Guidelines
- Determine whether the entities from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
- Write out the relationships of the entities in the target image. Your anwser should be based on the target image and the target image caption.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
- question: {{question}}
    - entities: {{entity1}} {{entity2}} ...
    - answer: {{answer}}
"""


RELATIONSHIP_ANSWER_TEMPLATE_ABLATION_2  = """# Your task
You are an assistant specialized in answering questions based on the content of images.

# Input data
1. Question input. The questions you should answer. The questions are focused on relationship attribute consistency of entities in the image. The questions are:
{questions}
2. The target image: <ImagePlaceholder>. You need to answer the questions based on the content of this image.

# Guidelines
Answer each question separately. For each question,
- Determine whether the entities from the question is in the target image. If yes, proceed to the next step. If no, just answer the entity does not exist in the iamge.
- Write out the relationships of the entities in the target image. Your anwser should be based on the target image and the target image caption.

# Output template
Repalce Variable in `{{}}`
Please generate your result based on following markdown template (Do NOT generate // comment in the template)
# Answer
- question: {{question}}
    - entities: {{entity1}} {{entity2}} ...
    - answer: {{answer}}
...
"""


INTRINSIC_EVAL_TEMPLATE = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate. The (question, answer) pair is focused on intrinsic attribute consistency of entities in the image. The answer is:
{answer}

# Guidelines
- Compare the answer with the structure information.
- If the answer is that the entity does not exist in the image, give a score of 0 for the answer. If not, proceed to the next step.
- Give a brief description of how well the answer matches the ground truth.
- Give a score from 0 to 10 for the answer according to the scoring strategy and the description above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
## {{entity name}}
- question: {{question}}
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
    - score: {{score}}
"""


INTRINSIC_EVAL_TEMPLATE_STAGE_1 = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate. The (question, answer) pair is focused on intrinsic attribute consistency of entities in the image. The answer is:
{answer}

# Guidelines
- Compare the answer with the structure information.
- Give a brief description of how well the answer matches the ground truth.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
## {{entity name}}
- question: {{question}}
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
"""


INTRINSIC_EVAL_TEMPLATE_STAGE_2 = """# Your task
You are an assistant specialized in grading the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate and its corresponding explanation. The (question, answer, explanation) triplet is focused on intrinsic attribute consistency of entities in the image. The answer and explanation is:
{answer_and_exp}
3. The target image: <ImagePlaceholder>. The question, answer and structure information are about this image.

# Guidelines
- Please give a score from 0 to 10 for the answer according to the following scoring strategy and the explanation above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0: The answer is that the entity does not exist in the image.
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


INTRINSIC_EVAL_TEMPLATE_ABLATION_2 = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate. The (question, answer) pair is focused on intrinsic attribute consistency of entities in the image. The answers are:
{answers}

# Guidelines
Evaluate each answer separately. For each answer,
- Compare the answer with the structure information.
- If the answer is that the entity does not exist in the image, give a score of 0 for the answer. If not, proceed to the next step.
- Give a brief description of how well the answer matches the ground truth.
- Give a score from 0 to 10 for the answer according to the scoring strategy and the description above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
## {{entity name}}
- question: {{question}}
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
    - score: {{score}}
...
"""


RELATIONSHIP_EVAL_TEMPLATE = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate. The (question, answer) pair is focused on relationship attribute consistency of entities in the image. The answer is:
{answer}

# Guidelines
- Compare the answer with the structure information.
- If the answer is that the entity does not exist in the image, give a score of 0 for the answer. If not, proceed to the next step.
- Give a brief description of how well the answer matches the ground truth.
- Give a score from 0 to 10 for the answer according to the scoring strategy and the description above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
- question: {{question}}
    - entities: {{entity1}} {{entity2}} ...
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
    - score: {{score}}
"""


RELATIONSHIP_EVAL_TEMPLATE_STAGE_1 = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate. The (question, answer) pair is focused on relationship attribute consistency of entities in the image. The answer is:
{answer}

# Guidelines
- Compare the answer with the structure information.
- Give a brief description of how well the answer matches the ground truth.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
- question: {{question}}
    - entities: {{entity1}} {{entity2}} ...
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
"""


RELATIONSHIP_EVAL_TEMPLATE_STAGE_2 = """# Your task
You are an assistant specialized in scoring the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answer you need to evaluate and its corresponding explanation. The (question, answer, explanation) triplet is focused on relationship attribute consistency of entities in the image. The answer and explanation is:
{answer_and_exp}
3. The target image: <ImagePlaceholder>. The question, answer and structure information are about this image.

# Guidelines
- Please give a score from 0 to 10 for the answer according to the following scoring strategy and the explanation above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0: The answer is that the entity does not exist in the image.
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


RELATIONSHIP_EVAL_TEMPLATE_ABLATION_2 = """# Your task
You are an expert in assessing the similarity between answers obtained from images and structure information (ground truth) obtained from text.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The answers you need to evaluate. The (question, answer) pair is focused on relationship attribute consistency of entities in the image. The answers are:
{answers}

# Guidelines
Evaluate each answer separately. For each answer,
- Compare the answer with the structure information.
- If the answer is that the entity does not exist in the image, give a score of 0 for the answer. If not, proceed to the next step.
- Give a brief description of how well the answer matches the ground truth.
- Give a score from 0 to 10 for the answer according to the scoring strategy and the description above. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0-3: The answer is not consistent with the structure information at all.
- 4-7: The answer is somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answer is very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)
# Evaluation
- question: {{question}}
    - entities: {{entity1}} {{entity2}} ...
    - answer: {{answer from the image}}
    - explanation: {{explanation}}
    - score: {{score}}
...
"""


OVERALL_SUMMARIZE_TEMPLATE = """# Your task
You are an expert in summarizing evaluation results of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of 3 parts: appearance quality, intrinsic attribute consistency and relationship attribute consistency. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results of appearance quality, give a summary about appearance quality, and give a score based on the summary. Score is from 0 to 10.
- Combining all the results of intrinsic attribute consistency, give a summary of intrinsic attribute consistency, and give a score based on the summary. Score is 0 to 10.
- Combining all the results of relationship attribute consistency, give a summary of relationship attribute consistency, and give a score based on the summary. Score is 0 to 10.
- Combining all the results of appearance quality, intrinsic attribute consistency, and relationship attribute consistency, give a description of the overall evaluation and give an overall score. Score is 0 to 10.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Appearance Quality Summary:
    - explanation: {{explanation}}
    - score: {{score}}
- Intrinsic Attribute Consistency Summary:
    - explanation: {{explanation}}
    - score: {{score}}
- Relationship Attribute Consistency Summary:
    - explanation: {{explanation}}
    - score: {{score}}
- Overall Score:
    - explanation: {{explanation}}
    - score: {{score}}
"""


OVERALL_SUMMARIZE_TEMPLATE_STAGE_1 = """# Your task
You are an expert in summarizing evaluation results of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of 3 parts: appearance quality, intrinsic attribute consistency and relationship attribute consistency. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results of appearance quality, give a summary about appearance quality.
- Combining all the results of intrinsic attribute consistency, give a summary of intrinsic attribute consistency.
- Combining all the results of relationship attribute consistency, give a summary of relationship attribute consistency.
- Combining all the results of appearance quality, intrinsic attribute consistency, and relationship attribute consistency, give a description of the overall evaluations.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Appearance Quality Summary:
    - explanation: {{explanation}}
- Intrinsic Attribute Consistency Summary:
    - explanation: {{explanation}}
- Relationship Attribute Consistency Summary:
    - explanation: {{explanation}}
- Overall Score:
    - explanation: {{explanation}}
"""


OVERALL_SUMMARIZE_TEMPLATE_STAGE_2 = """# Your task
You are an expert in scoring the quality of generated image according to evaluation results of the image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of 3 parts: appearance quality, intrinsic attribute consistency and relationship attribute consistency. The evaluation result is:
{eval_result_and_exp}
3. The target image: <ImagePlaceholder>.

# Scoring strategy: Overall scoring
- Combining all the results of appearance quality and the corresponding explanation, and give a score from 0 to 10.
- Combining all the results of intrinsic attribute consistency and the corresponding explanation, and give a score from 0 to 10.
- Combining all the results of relationship attribute consistency and the corresponding explanation, and give a score from 0 to 10.
- Combining all the results of appearance quality, intrinsic attribute consistency, and relationship attribute consistency and their explanations, and give an Overall Score from 0 to 10.

# Output format
Please output your 4 scores in one line, namely appearance quality Score, intrinsic attribute consistency Score, relationship attribute consistency Score and Overall Score, separated by spaces between the scores. Be careful not to generate any content other than the scores.

# Scores
"""


APPEARANCE_SUMMARIZE_TEMPLATE = """# Your task
You are an expert in summarizing evaluation results about appearance quality of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions about appearance quality of the image. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results, give a summary for the appearance quality of the image, and give a score based on the summary. Score is from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy    
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Appearance Quality Summary:
    - explanation: {{explanation}}
    - score: {{score}}
"""


APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_1 = """# Your task
You are an expert in summarizing evaluation results about appearance quality of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions about appearance quality of the image. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results, give a summary for the appearance quality of the image.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Appearance Quality Summary:
    - explanation: {{explanation}}
"""


APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_2 = """# Your task
You are an expert in scoring the appearance quality of generated image according to evaluation results of the image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions and an overall evaluation about appearance quality of the image. The evaluation result is:
{eval_result_and_exp}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results and the corresponding explanation, and give a score from 0 to 10. Score is from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy
- 0-3: The appearance is not realistic, aesthetically pleasing, or align with human intuition at all.
- 4-7: The appearance is somewhat realistic, aesthetically pleasing, or align with human intuition.
- 8-10: The appearance is very realistic, aesthetically pleasing, or align with human intuition.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


INTRINSIC_SUMMARIZE_TEMPLATE = """# Your task
You are an expert in summarizing evaluation results about intrinsic attribute consistency of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions about intrinsic attribute consistency of the image. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results, give a summary for intrinsic attribute consistency of the image, and give a score based on the summary. Score is from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy    
- 0: The entities does not exist in the image.
- 0-3: The answers are not consistent with the structure information at all.
- 4-7: The answers are somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answers are very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Intrinsic Attribute Consistency Summary:
    - explanation: {{explanation}}
    - score: {{score}}
"""


INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_1 = """# Your task
You are an expert in summarizing evaluation results about intrinsic attribute consistency of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions about intrinsic attribute consistency of the image. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results, give a summary for intrinsic attribute consistency of the image.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Intrinsic Attribute Consistency Summary:
    - explanation: {{explanation}}
"""


INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_2 = """# Your task
You are an expert in scoring the intrinsic attribute consistency of generated image according to evaluation results of the image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions and an overall evaluation about intrinsic attribute consistency of the image. The evaluation result is:
{eval_result_and_exp}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results and the corresponding explanation, and give a score for the intrinsic attribute consistency of the image from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Scoring strategy  
- 0: The entities does not exist in the image.
- 0-3: The answers are not consistent with the structure information at all.
- 4-7: The answers are somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answers are very consistent with the structure information.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


RELATIONSHIP_SUMMARIZE_TEMPLATE = """# Your task
You are an expert in summarizing evaluation results about relationship attribute consistency of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions about relationship attribute consistency of the image. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results, give a summary for relationship attribute consistency of the image, and give a score based on the summary. Score is from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.
    
# Scoring strategy
- 0: The entities does not exist in the image.
- 0-3: The answers are not consistent with the structure information at all.
- 4-7: The answers are somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answers are very consistent with the structure information.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Relationship Attribute Consistency Summary:
    - explanation: {{explanation}}
    - score: {{score}}
"""


RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_1 = """# Your task
You are an expert in summarizing evaluation results about relationship attribute consistency of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions about relationship attribute consistency of the image. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results, give a summary for relationship attribute consistency of the image.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Relationship Attribute Consistency Summary:
    - explanation: {{explanation}}
"""


RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_2 = """# Your task
You are an expert in scoring the relationship attribute consistency of generated image according to evaluation results of the image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of multiple questions and an overall evaluation about relationship attribute consistency of the image. The evaluation result is:
{eval_result_and_exp}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results and the corresponding explanation, and give a score for the relationship attribute consistency of the image from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.
    
# Scoring strategy 
- 0: The entities does not exist in the image.
- 0-3: The answers are not consistent with the structure information at all.
- 4-7: The answers are somewhat consistent with the structure information. Semantics are similar but not entirely consistent.
- 8-10: The answers are very consistent with the structure information.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""


MERGE_SUMMARIZE_TEMPLATE = """# Your task
You are an expert in summarizing evaluation results of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result and summary you need to summarize. The evaluation result and summary consists of 3 parts: appearance quality, intrinsic attribute consistency and relationship attribute consistency. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the results of appearance quality, intrinsic attribute consistency, and relationship attribute consistency, give a description of the overall evaluation and give an overall score. Score is from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Overall Score:
    - explanation: {{explanation}}
    - score: {{score}}
"""


MERGE_SUMMARIZE_TEMPLATE_STAGE_1 = """# Your task
You are an expert in summarizing evaluation results of generated image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation result you need to summarize. The evaluation result consists of 3 parts: appearance quality, intrinsic attribute consistency and relationship attribute consistency. The evaluation result is:
{eval_result}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the evaluation results, summaries and scores of appearance quality, intrinsic attribute consistency and relationship attribute consistency, give a description of the overall evaluations.

# Output template
Repalce Variable in `{{}}`
Please generate your output based on following markdown template (Do NOT generate // comment in the template)

- Overall Score:
    - explanation: {{explanation}}
"""


MERGE_SUMMARIZE_TEMPLATE_STAGE_2 = """# Your task
You are an expert in scoring the quality of generated image according to evaluation results and summaries of the image.

# Input data
1. The structure information extracted from the image:
{structure_info}
2. The evaluation results, summaries and scores. The results consists of 3 parts: appearance quality, intrinsic attribute consistency and relationship attribute consistency. The results are:
{eval_result_and_exp}
3. The target image: <ImagePlaceholder>.

# Guidelines
- Combining all the evalution results, summaries and scores of appearance quality, intrinsic attribute consistency, and relationship attribute consistency and their explanations, and give an Overall Score from 0 to 10. You should be as strict as possible in your scoring. The lower the score, the less the answer matches the ground truth.

# Output format
Please generate your score on a single line, be careful not to generate any content other than the score.

# Score
"""
