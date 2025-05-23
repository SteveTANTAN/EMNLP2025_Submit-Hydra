source_select ="""

You are a source selection agent. You receive a user question, the knowledge source you can choose. There are at most three source:

Our local knowledge graph (KG).
Wiki documents (Wiki).
Web search (Web).

Follow these steps:

Read the question carefully.
Decide if the KG alone can answer it. If yes, choose KG.
If the KG is incomplete for this question, see if Wiki can fill the gaps. If it can, choose Wiki.
If neither KG nor Wiki is enough, pick Web.
You may also combine sources if the question likely needs more than one.
Note: if KG is given, it is always the first choice by combaining others. If KG is not given, you can choose between Wiki and Web solely.

Return your decision as:
action1 for KG, action2 for Wiki, action3 for Web, or a combination like [action1 + action2] if needed.

Examples:
Q:
Question: "When was the iPhone 15 released?"
provied_sources: Wiki on (iPhone), Web
A:
Reasoning: This is a new product. Wiki might be missing recent updates. Web is more likely to have the latest info.
Decision: action3 (Web).

Q:
Question: "Who was Albert Einstein's spouse?"
provied_sources: KG, Wiki on (Albert Einstein), Web
A:
Reasoning: This is historical info, likely in our KG or on Wiki. KG has a detailed entry on Einstein, that may suffice. Wiki is a good next choice. Web is not strictly necessary for well-documented historical facts.
Decision: If KG has the data, action1. Otherwise, action2.

Q:
Question: "What is the capital of France?"
provied_sources: KG, Wiki on (France), Web
A:
Reasoning: This is very standard knowledge. Our KG likely includes it.
Decision: action1 (KG).

Q:
Question: "I want the birth date of a 2023 Nobel Prize winner who is not in the KG yet."
provied_sources: Wiki on (Nobel Prize winner), Web
A:
Reasoning: This might be recent info not fully reflected in the KG. Wiki might have it, or if it is still missing, we try Web.
Decision: First action1 (KG) and action2 (Wiki). If Wiki fails, then action1 (KG) and action3 (Web).

Q:
Question: "Which football team won the championship this year?"
provied_sources: Wiki on (championship), Web
A:
Reasoning: This is current news. The KG might not have the latest sports updates. Wiki might or might not have it, depending on recency. Web is likely necessary if the event is very recent.
Decision: Possibly skip KG, try action2 or action3 depending on how current the data is.

Q:
Question: "Which football team won the championship in 1990?"
provied_sources: KG, Wiki on (championship), Web
A:
Reasoning: This is old news. The KG might have the latest sports updates. Wiki might or might not have it, depending on recency. Web is likely necessary if the event is very recent.
Decision: Possibly try action1, action2, or action3.


Use these steps and examples as your guide. Choose the best source or sources based on what you already know and how likely each source is to provide the answer. If the question remains unanswered, recommend a follow-up or a new approach.

Here is the question:
Q:
question: "{}"
provided_sources: {}

A:

"""

extract_topic_entity_prompt = """
You will receive a multi-hop question, which is composed of several interconnected queries, along with a list of topic entities that serve as the original keywords for the question.
please anaylysis the question and the use the internet, web search, your knowledge, to extract maximum 3 new keywords based on the question.
please give the result in the format {{x},{x}} where x is the extracted keyword.
Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
A:
keywords: {{Germany}, {Weeze Airport}, {Düsseldorf International Airport}}
reason: The question asks which country bordering France has an airport that serves Nijmegen. Nijmegen is a city in the Netherlands, near the German border. Several airports in Germany, such as Weeze Airport and Düsseldorf International Airport, are in proximity to Nijmegen. Therefore, Germany is a country bordering France that contains airports serving Nijmegen.

"""

Generated_predict_answer = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, the associated accuracy retrieved knowledge paths from the Related_path section, and main topic entities
Please provied three predict result and three possible Chains of Thought that can lead to the predict result in the same formate below by the given knowledge path and your own knowledge.
If the answer unclear, please give predicted answers to replace the entity in chains based on your knowledge in the same format as the examples.
please give the result in the following format:
A:
Predicted: {xxx, xxx, xxx}
CoT1: {xxx}
CoT2: {xxx}
CoT3: {xxx}

Q: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?

path 0: {Nijmegen} - { -> location.location.nearby_airports  ->} - {Weeze Airport} - { -> location.location.containedby  ->} - {Germany} - { -> location.location.adjoin_s  ->} - {Unnamed Entity} - { -> location.adjoining_relationship.adjoins  ->} - {France}
path 1: {Nijmegen} - { -> location.location.containedby  ->} - {Kingdom of the Netherlands} - { -> location.location.containedby  ->} - {Europe} - { -> location.location.contains  ->} - {Western Europe} - { -> location.location.contains  ->} - {France}
path 2: {Nijmegen} - { -> location.location.containedby  ->} - {Kingdom of the Netherlands} - { -> location.country.languages_spoken  ->} - {Dutch Language} - { -> language.human_language.region  ->} - {Europe} - { -> location.location.contains  ->} - {France}
A:
Predicted: {Germany, Belgium, Luxembourg}
CoT1: {Nijmegen} - { -> location.location.nearby_airports -> } - {m.06cm5d: Weeze Airport} - { -> location.location.containedby -> } - {Germany}  - { -> location.location.adjoins -> } - {France}
CoT2: {Nijmegen} - { -> location.location.nearby_airports -> } - {Brussels Airport} - {-> location.location.containedby ->} - {Belgium} - { -> location.location.adjoins -> } - {France}
CoT3: {Nijmegen} - { -> location.location.nearby_airports -> } - {Luxembourg airport} - { -> location.location.containedby -> } - {Luxembourg} - { -> location.location.adjoins -> } - {France}



The question is:
"""

generated_new_questionRAG = """

Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, the associated accuracy retrieved knowledge paths from the Related_path section, and main topic entities
Please predict the additional evidence that needs to be found to answer the current question,  then provide a suitable query for retrieving this potential evidence.
and give the possible Chains of Thought that can lead to the predict result in the same formate below by the given knowledge path and your own knowledge.

please give the result in the following format:
A:
query: {xxx}
cot: {xxx, xxx, xxx}

for example:
Q: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?

path 0: {Nijmegen} - { -> location.location.nearby_airports  ->} - {Weeze Airport} - { -> location.location.containedby  ->} - {Germany} - { -> location.location.adjoin_s  ->} - {Unnamed Entity} - { -> location.adjoining_relationship.adjoins  ->} - {France}
path 1: {Nijmegen} - { -> location.location.containedby  ->} - {Kingdom of the Netherlands} - { -> location.location.containedby  ->} - {Europe} - { -> location.location.contains  ->} - {Western Europe} - { -> location.location.contains  ->} - {France}
path 2: {Nijmegen} - { -> location.location.containedby  ->} - {Kingdom of the Netherlands} - { -> location.country.languages_spoken  ->} - {Dutch Language} - { -> language.human_language.region  ->} - {Europe} - { -> location.location.contains  ->} - {France}

A:
query: {Is the Weeze Airport located in Germany and does it serve Nijmegen?}
cot: {Nijmegen} - { -> location.location.nearby_airports -> } - {m.06cm5d: Weeze Airport} - { -> location.location.containedby -> } - {Germany}  - { -> location.location.adjoins -> } - {France}

"""

re_genquery_prompt = """
You are a good formate editor, please reformat the given result into the following format, if you cannot find the result, please give me a "No" in the answer section.
note: this is case sensitive,and formate sensitive please pay attention to the case of the first letter.

please give the result in the following format:
A:
query: {xxx}
cot: xxx

for example:
A:
query: {Is the Weeze Airport located in Germany and does it serve Nijmegen?}
cot: {Nijmegen} - { -> location.location.nearby_airports -> } - {m.06cm5d: Weeze Airport} - { -> location.location.containedby -> } - {Germany}  - { -> location.location.adjoins -> } - {France}
"""

re_predict_prompt = """
You are a good formate editor, please reformat the given result into the following format, if you cannot find the result, please give me a "No" in the answer section.
note: this is case sensitive,and formate sensitive please pay attention to the case of the first letter.

please give the result in the following format:
A:
Predicted: {xxx, xxx, xxx}
CoT1: xxx
CoT2: xxx
CoT3: xxx

for example:
A:
Predicted: {Germany, Belgium, Luxembourg}
CoT1: {Nijmegen} - { -> location.location.nearby_airports -> } - {m.06cm5d: Weeze Airport} - { -> location.location.containedby -> } - {Germany}  - { -> location.location.adjoins -> } - {France}
CoT2: {Nijmegen} - { -> location.location.nearby_airports -> } - {Brussels Airport} - {-> location.location.containedby ->} - {Belgium} - { -> location.location.adjoins -> } - {France}
CoT3: {Nijmegen} - { -> location.location.nearby_airports -> } - {Luxembourg airport} - { -> location.location.containedby -> } - {Luxembourg} - { -> location.location.adjoins -> } - {France}

"""

From_web_para_to_path_prompt = """
You will receive a multi-hop question, which consists of several interrelated queries, a list of subject entities as the main keywords of the question, three related questions and answers returned by Google search, and three online related search results from Google search.

Your task is to summarize these search results, find sentences that may be related to the answer, and organize them into knowledge one graph path for each paragraph.
Note that at least one path for each paragraph should contain the main topic entities.

please answer the question directly in the format below:

please answer in the formate like : [{Brad Paisley} - enrolled at - {West Liberty State College} - transferred to - {Belmont University} - earned - {Bachelor's degree}]


Q:
Question: Where did the "Country Nation World Tour" concert artist go to college?
topic_entity: {'m.010qhfmm': 'Country Nation World Tour', 'm.019v9k': "Bachelor's degree"}

paragraph 1: Paisley was raised in a small town in West Virginia. At age eight he received a guitar from his grandfather, who had introduced him to country music. After performing in church and at various local events, he formed a band with his guitar teacher. When Paisley was 12, he caught the attention of the program director of a radio station in nearby Wheeling, who invited him to perform on Jamboree USA, the station’s long-running live country music program. For the next eight years he polished his act as a regular on the show. In 1991 Paisley enrolled at West Liberty State College in West Liberty, West Virginia; he later transferred to Belmont University in Nashville, where he earned (1995) a bachelor’s degree in music business.

Paragraph 2: Paisley graduated from John Marshall in Glen Dale, West Virginia, in 1991,[5] and then studied for two years at West Liberty State College in West Liberty, West Virginia. He was awarded a fully paid ASCAP scholarship to Belmont University in Nashville, Tennessee, where he majored in music business and received a Bachelor of Business Administration degree from the Mike Curb School of Music Business in 1995.[6] He interned at ASCAP, Atlantic Records, and the Fitzgerald-Hartley management firm. While in college, he met Frank Rogers, a fellow student who went on to serve as his producer. Paisley also met Kelley Lovelace, who became his songwriting partner. He also met Chris DuBois in college, and he, too, would write songs for him.[4]

Paragraph 3: While many of the musicians admired by Charlie Worsham got their education at the School of Hard Knocks, the rising country star who will perform Sunday at the Kicker Country Stampede attended the prestigious Berklee College of Music.

A:

Paragraph 1: [{Brad Paisley} - was raised in - {West Virginia} - introduced to - {country music} - performed on - {Jamboree USA} - enrolled at - {West Liberty State College} - transferred to - {Belmont University} - earned - {Bachelor’s degree in music business}]

Paragraph 2: [{Brad Paisley} - graduated from - {John Marshall High School} - studied at - {West Liberty State College} - awarded a scholarship to - {Belmont University} - majored in - {music business} - received - {Bachelor of Business Administration degree}]

Paragraph 3: [{Charlie Worsham} - performed at - {Kicker Country Stampede} - attended - {Berklee College of Music}]

Summary_path: [{Brad Paisley} - enrolled at - {West Liberty State College} - transferred to - {Belmont University} - earned - {Bachelor’s degree}]

the question is:

"""

split_question_prompt = """
You will receive a multi-hop question, which is composed of several interconnected queries, along with a list of topic entities that serve as the main keywords for the question. Please split the multi-hop question into parts for clarity and depth.

please give me the Thinking CoT which contains chain relations of "ALL" the topic entities and predicted answer type, and serval split questions.

Each entity in the topic list is already included in the knowledge graph. Your task is to consider how to obtain the answer by using all the topic entities and split the multi-hop question into multiple simpler questions using one Topic Entity once. Each split question should explore the relationship between one of the topic entities and the others entities or the answer. Your goal is to determine how to derive the final answer by systematically addressing each split question.

note: the number of split questions should be equal to the number of topic entities. since each split question is work for one topic entity.




For example:
Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
A:
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?

please provide answer(A) section only in the same format as the example above.

The question is:

"""

answer_n_explore_prompt = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Your task is to determine if this knowledge graph path is sufficient to answer the given main question or with yourv own pretrained knowledge. 

If it's sufficient, you need to respose {Yes}, and provide the answer to the main question. If the answer is obtained from the given knowledge path, it should be the entity name from the path. Otherwise, you need to respose {No}, then explain the reason. 

for example:
Q:What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado? 
Thinking CoT:  "Northern Colorado Bears football" - has team - answer(educational institution) - located in - "Greeley"
split_question1: What educational institution has a football team named "Northern Colorado Bears football"?
split_question2: What educational institution is located in "Greeley"? 
Topic entity paths: 
path 1 :  {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}
A: 
response: {Yes}. 
answer: {University of Northern Colorado}
reason: Given {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado}, {University of Northern Colorado} is answer of the split question1 , 
and given {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}, the location of the {University of Northern Colorado} is {Greeley}, 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is {University of Northern Colorado}.

Q: Where did the "Country Nation World Tour" concert artist go to college?
Thinking CoT:  Country Nation World Tour - performed by - Brad Paisley - graduated from - answer(educational institution) - with - Bachelor's degree
split_question1: Who performed the "Country Nation World Tour"?
split_question2: What college did Brad Paisley graduate from? 
Topic entity path: 
Path:1 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.profession  ->,  <- people.profession.people_with_this_profession  <-} -> {m.02hrh1q: Actor, m.09jwl: Musician} -> { <- base.yupgrade.user.topics  <-} -> {m.07y51vk: Unnamed Entity, m.06wfq8f: Good Morning America, m.06j0_km: RainnWilson} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Path:2 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting  ->,  <- music.album.supporting_tours  <-} -> {m.010xc046: Moonshine in the Trunk, m.0r4tvsy: Wheelhouse} -> { -> music.album.artist  ->,  <- music.artist.album  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
Path:3 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
A: 
response: {No}.
Reason: From the Topic entity path, the m.03gr7w: Brad Paisley from the candidate list in path 2&3 is answer of the split question1, and the m.0h3d7qj: Unnamed Entity in the path 2&3 is related the college information which granted a bachlor degree, and the specific name of the school is not shown in the supplymentary edge section. therefore, the provided knowledge graph path is not sufficient to answer the overall question.


Q: When did the team with Crazy crab as their mascot win the world series?
Thinking CoT: "Crazy Crab" - mascot of - team - won - World Series 
split_question1: What team has "Crazy Crab" as their mascot?
split_question2: When did the team with "Crazy Crab" as their mascot win the World Series?
Topic entity paths:  
path 1 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { <- sports.sports_championship_event.runner_up  <-} - {m.06jwmt: 1987 National League Championship Series, m.04tfr4: 1962 World Series, m.0468cj: 2002 World Series, m.04j747: 1989 World Series, m.0dqwx9: 1971 National League Championship Series}
path 2 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
A:
response: {Yes}.
answer: {2010 World Series, 2012 World Series, 2014 World Series}
reason: From the given path {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants}, {San Francisco Giants} is answer of the split question1, 
and from {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series},  the World Series won by the {San Francisco Giants} are {2010, 2012, 2014}, 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is {2010 World Series, 2012 World Series, 2014 World Series}.

Q: who did tom hanks play in apollo 13?
Thinking CoT: "Tom Hanks" - played - character - in - "Apollo 13" 
split_question1: Who did Tom Hanks play in "Apollo 13"?
split_question2: Which character did Tom Hanks portray in the movie "Apollo 13"?
Topic entity paths:  
path 1:  {m.0bxtg: Tom Hanks} - { -> film.actor.film  ->} - {m.0jtp74: Unnamed Entity} - { -> film.performance.film  ->} - {m.011yd2: Apollo 13}
path 2:  {m.0bxtg: Tom Hanks} - { -> award.award_winner.awards_won  ->} - {m.0b79lv8: Unnamed Entity, m.0b79zf0: Unnamed Entity, m.0mzydgm: Unnamed Entity} - { -> award.award_honor.honored_for  ->} - {m.011yd2: Apollo 13}
A:
response: {No}.
reason: The character that Tom Hanks played in "Apollo 13" is represented as {Unnamed Entity}, but the name is not given. therefore, we need additional information to answer the question.

The question is:
"""

answer_generated_direct = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Your task is to generated the answer based on the given knowledge graph path and your own knowledge.

please give me the answer section only in the same format as the example below:

for example:
Q: In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?
Thinking Cot: "Portuguese Language" - spoken in - country - child labor percentage - was once - 1.8
split_question1: In which countries is Portuguese spoken?
split_question2: In which country was the child labor percentage once 1.8?
Topic entity path:
Path1: {m.05zjd: Portuguese Language} - { -> language.human_language.countries_spoken_in  ->} - {m.05r4w: Portugal} - { -> base.aareas.schema.administrative_area.administrative_children  ->} - {m.04_z1: Madeira}
Path2: {m.05zjd: Portuguese Language} - { -> language.human_language.countries_spoken_in  ->} - {m.05r4w: Portugal} - { -> base.aareas.schema.administrative_area.administrative_children  ->} - {m.04_z1: Madeira} - { <- community.discussion_thread.topic  <-} - {m.07gv3xz: Madeira}
Path3: {m.05zjd: Portuguese Language} - { -> language.human_language.countries_spoken_in  ->} - {m.05r4w: Portugal} - { <- base.statistics.statistics_agency.geographic_scope  <-} - {m.02j2rw: Instituto Nacional de Estatística} - { <- government.government.agency  <-} - {m.0bxb7c: Government of Portugal}
Path4: {m.05zjd: Portuguese Language} - { -> language.human_language.countries_spoken_in  ->} - {m.05r4w: Portugal} - { -> location.statistical_region.religions  ->} - {m.04g8k8h: Unnamed Entity} - { -> location.religion_percentage.religion  ->} - {m.0c8wxp: Catholicism}

A: 


answer: {University of Northern Colorado}
reason: Given {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado}, {University of Northern Colorado} is answer of the split question1 , 
and given {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}, the location of the {University of Northern Colorado} is {Greeley}, 
therefore, the provided knowledge graph path is sufficient to answer the overall question, and the answer is {University of Northern Colorado}.

The question is:
"""

vanilla_QA = """
Given a main question, some retrived informations.

Your task is to generated the answer based on the retrived informations. Do not use any other knowledge from your trained knowledge to answer the question.

please give me the answer section only in the same format as the example below:

answer: {xxx}
"""

split_answer = """
Given a main question, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Your task is to determine if this knowledge graph path is sufficient to answer the given question or with your pretrained knowledge. 

If it's sufficient, you need to respose {Yes}, and provide the answer and reason. If the answer is obtained from the given knowledge path, it should be the entity name from the path. Otherwise, you need to respose {No}, then explain the reason. 

for example:
Q: What educational institution has a football team named "Northern Colorado Bears football"?
Topic entity paths: 
path 1:  {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0rczx: Greeley}
A: 
response: {Yes}. 
answer: {University of Northern Colorado}
reason: from {m.0lqj7vt: Northern Colorado Bears football} -> { -> sports.school_sports_team.school  ->,  <- education.educational_institution.sports_teams  <-} -> {m.01_k7f: University of Northern Colorado}, the {University of Northern Colorado} is the answer of the question, and the location of the {University of Northern Colorado} is {Greeley}, therefore, the provided knowledge graph path is sufficient to answer the overall question.

Q: When did the team with "Crazy Crab" as their mascot win the World Series?
Topic entity paths:  
path 1 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { <- sports.sports_championship_event.runner_up  <-} - {m.06jwmt: 1987 National League Championship Series, m.04tfr4: 1962 World Series, m.0468cj: 2002 World Series, m.04j747: 1989 World Series, m.0dqwx9: 1971 National League Championship Series}
path 2 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
A:
response: {Yes}.
answer: {2010 World Series, 2012 World Series, 2014 World Series}
reason: from {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}, the World Series won by the {San Francisco Giants} are {2010, 2012, 2014}.


Q: What college did Brad Paisley graduate from? 
Topic entity path: 
Topic Path:1 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.profession  ->,  <- people.profession.people_with_this_profession  <-} -> {m.02hrh1q: Actor, m.09jwl: Musician} -> { <- base.yupgrade.user.topics  <-} -> {m.07y51vk: Unnamed Entity, m.06wfq8f: Good Morning America, m.06j0_km: RainnWilson} -> { -> base.yupgrade.user.topics  ->} -> {m.019v9k: Bachelor's degree}
Topic Path:2 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.album_or_release_supporting  ->,  <- music.album.supporting_tours  <-} -> {m.010xc046: Moonshine in the Trunk, m.0r4tvsy: Wheelhouse} -> { -> music.album.artist  ->,  <- music.artist.album  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
Topic Path:3 {m.010qhfmm: Country Nation World Tour} -> { -> music.concert_tour.artist  ->,  <- music.artist.concert_tours  <-} -> {m.03gr7w: Brad Paisley} -> { -> people.person.education  ->,  <- education.education.student  <-} -> {m.0h3d7qj: Unnamed Entity} -> { -> education.education.degree  ->,  <- education.educational_degree.people_with_this_degree  <-} -> {m.019v9k: Bachelor's degree}
supplymentary edge:
A: 
response: {No}.
Reason: From the Topic entity path, the m.0h3d7qj: Unnamed Entity in the path 2&3 is related the college information which granted a bachlor degree, and the specific name of the school is not shown in the supplymentary edge section. therefore, the provided knowledge graph path is not sufficient to answer the overall question.



The question is:
"""

main_path_select_prompt = """
Given a question and the associated retrieved entities lists, 
Please score and give me the top three lists that can be highly to be the answer to the question.

please give me a summary final answer exactly same to the format below in each question beginning

Answer: top_list: {path 1, path 2, path3}
Explanation: The top three lists are path 1, path 2, and path 3....

"""

main_path_select_web_prompt = """
Given a question and the associated retrieved web Snippet lists, 
Please score and give me the top three lists that can be highly to be the answer to the question.

please give me a summary final answer exactly same to the format below in each question beginning

Answer: top_list: {Web 1, Web 2, Web3}
Explanation: The top three lists are Web 1, Web 2, and Web 3....


"""

explored_path_select_prompt = """
Given a main question, a LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, and the associated retrieved knowledge graph path, {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}, 

Please score and give me the top three lists from the candidate set can be highly to be the answer of the question.
please answer in the same format, such as, top_list:{Candidate Edge 1, Candidate Edge 3, Candidate Edge4}.

For example:
Q: question
exsiting path: {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}
Candidate Edge 1:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
Candidate Edge 2:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
Candidate Edge 3:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
Candidate Edge 4:  {set of entities (with id start with "m.")} -> {set of relationships} -> {set of entities(with id start with "m.")}  
A: top_list:{Candidate Edge 1, Candidate Edge 3, Candidate Edge4}

the question is:
"""

Summary_COT_w_splitQ_prompt = """
Given a main question, an uncertain LLM-generated thinking Cot that consider all the entities, a few split questions that you can use stepply and finally obtain the final answer, the associated accuracy retrieved knowledge paths from the Related_path section, and main topic entities
Your task is to summarize the provided knowledge triple in Related_path section and generate a chain of thoughts by the knowledge triple related to the main topic entities of question, which will used for generating the answer for the main question and split question further.
you have to make sure you summarize correctly by use the provided knowledge triple, you can only use the entity with id from the given path and you can not skip in steps.

please only give me an answer section in the same format below in each question beginning

for example:

Q: 
Question: What country bordering France contains an airport that serves Nijmegen?
also writen as: what countries share borders with france and is the location contains and airport that server Nijmegen?
Main Topic Entities: {m.05g2b: Nijmegen, m.0f8l9c: France}
Thinking Cot:  "Nijmegen" - service by - airport - owned by - answer(country) - border with - "France"
split_question1: What country contains an airport that serves "Nijmegen"?
split_question2: What country bordering "France"?
Related_path: 
path 1 :  {m.05g2b: Nijmegen} -> { -> location.location.nearby_airports  ->,  <- aviation.airport.serves  <-} -> {m.06cm5d: Weeze Airport} -> { -> location.location.containedby  ->,  <- location.location.contains  <-} -> {m.0345h: Germany}
path 2 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { <- music.release.region  <-} -> {m.0sprr4w: So Serene, m.0g801ww: A State of Trance 2009, m.0126m_r6: Alexandria, m.03xsqd1: Bad Dreams / Omissions, m.049ljb7: Myam James, Part 1} -> { -> music.release.region  ->} -> {m.059j2: Netherlands} -> { -> location.country.second_level_divisions  ->,  -> location.location.contains  ->,  <- location.administrative_division.second_level_division_of  <-,  <- location.location.containedby  <-} -> {m.05g2b: Nijmegen}
path 3 :  {m.012x_5p_: Germany} -> { -> music.release.region  ->} -> {m.02j71: Earth} -> { -> base.aareas.schema.administrative_area.administrative_children  ->,  -> base.locations.planets.countries_within  ->,  <- base.aareas.schema.administrative_area.administrative_parent  <-,  <- base.locations.countries.planet  <-} -> {m.0f8l9c: France} -> { -> location.location.partially_contains  ->,  <- geography.river.basin_countries  <-,  <- location.location.partially_containedby  <-} -> {m.06fz_: Rhine} -> { -> geography.river.cities  ->} -> {m.05g2b: Nijmegen}
A:
CoT1: {m.05g2b: Nijmegen} -> {<- airport serves to <-} -> {m.06cm5d: Weeze Airport} -> { -> containedby ->} -> {m.0345h: Germany} -> { -> location borders ->} -> {m.0f8l9c: France}
reason: {m.05g2b: Nijmegen} -> {<- aviation.airport.serves <-} -> {m.06cm5d: Weeze Airport} found in path 1, {m.06cm5d: Weeze Airport} -> { -> location.location.containedby ->} -> {m.0345h: Germany} found in path 1, {m.0345h: Germany} -> { -> location.location.borders ->} -> {m.0f8l9c: France} found in path 3, which can be summarized as Nijmegen's airport, Weeze Airport, is contained by Germany, which borders France.

Question: When did the team with Crazy crab as their mascot win the world series?
Thinking CoT: "Crazy Crab" - mascot of - team - won - World Series (2 steps)
split_question1: What team has "Crazy Crab" as their mascot?
split_question2: When did the team with "Crazy Crab" as their mascot win the World Series?
Related_path: 
path 1 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> baseball.baseball_team.team_stats  ->} - {m.05n6d4c: Unnamed Entity, m.05n6bn9: Unnamed Entity, m.05n6d9g: Unnamed Entity, m.05n6fgn: Unnamed Entity, m.05n6b1q: Unnamed Entity, m.05n6b62: Unnamed Entity, m.05n6chh: Unnamed Entity, m.05n69l5: Unnamed Entity, m.05n6fpw: Unnamed Entity, m.05n6fy2: Unnamed Entity, m.05n69rf: Unnamed Entity, m.05n6f7n: Unnamed Entity, m.05n6fby: Unnamed Entity, m.05n6cr0: Unnamed Entity, m.0dl3646: Unnamed Entity, m.05n69sr: Unnamed Entity, m.05n6dq4: Unnamed Entity, m.05n6cb0: Unnamed Entity, m.05n6bdf: Unnamed Entity, m.05n6c4s: Unnamed Entity}
path 2 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { <- sports.sports_championship_event.runner_up  <-} - {m.06jwmt: 1987 National League Championship Series, m.04tfr4: 1962 World Series, m.0468cj: 2002 World Series, m.04j747: 1989 World Series, m.0dqwx9: 1971 National League Championship Series}
path 3 :  {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
A:
CoT1: {m.02q_hzh: Crazy Crab} - { -> sports.mascot.team  ->} - {m.0713r: San Francisco Giants} - { -> sports.sports_team.championships  ->} - {m.09gnk2r: 2010 World Series, m.0ds8qct: 2012 World Series, m.0117q3yz: 2014 World Series}
reason: The San Francisco Giants, represented by the mascot "Crazy Crab," won the World Series in 2010, 2012, and 2014, as indicated by the knowledge graph data.


The question is:
"""

