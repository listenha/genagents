Generative Agent Simulations of 1,000 People
Authors: Joon Sung Park1*, Carolyn Q. Zou1,2, Aaron Shaw2, Benjamin Mako Hill3, Carrie Cai4,
Meredith Ringel Morris5, Robb Willer6, Percy Liang1, Michael S. Bernstein1
Affiliations:
1Computer Science Department, Stanford University; Stanford, CA, 94305, USA.
2Department of Communication Studies, Northwestern University; Evanston, IL, 60208,
USA.
3Department of Communication, University of Washington; Seattle, WA 98195, USA.
4Google DeepMind; Mountain View, CA 94043, USA.
5Google DeepMind; Seattle, WA 98195, USA.
6Department of Sociology, Stanford University; Stanford, CA, 94305, USA.
*Corresponding author. Email: joonspk@stanford.edu
Abstract:
The promise of human behavioral simulation—general-purpose computational agents that
replicate human behavior across domains—could enable broad applications in policymaking and
social science. We present a novel agent architecture that simulates the attitudes and behaviors of
1,052 real individuals—applying large language models to qualitative interviews about their
lives, then measuring how well these agents replicate the attitudes and behaviors of the
individuals that they represent. The generative agents replicate participants' responses on the
General Social Survey 85% as accurately as participants replicate their own answers two weeks
later, and perform comparably in predicting personality traits and outcomes in experimental
replications. Our architecture reduces accuracy biases across racial and ideological groups
compared to agents given demographic descriptions. This work provides a foundation for new
tools that can help investigate individual and collective behavior.
1





Main Text: General-purpose simulation of human attitudes and behavior—where each simulated
person can engage across a range of social, political, or informational contexts—could enable a
laboratory for researchers to test a broad set of interventions and theories (1-3). How might, for
instance, a diverse set of individuals respond to new public health policies and messages, react to
product launches, or respond to major shocks? When simulated individuals are combined into
collectives, these simulations could help pilot interventions, develop complex theories capturing
nuanced causal and contextual interactions, and expand our understanding of structures like
institutions and networks across domains such as economics (4), sociology (2), organizations (5),
and political science (6).
Simulations define models of individuals that are referred to as agents (7). Traditional agent
architectures typically rely on manually specified behaviors, as seen in agent-based models (1, 8,
9), game theory (10), and discrete choice models (11), prioritizing interpretability at the cost of
restricting agents to narrow contexts and oversimplifying the contingencies of real human
behavior (3, 4). Generative artificial intelligence (AI) models, particularly large language models
(LLMs) that encapsulate broad knowledge of human behavior (12-15), offer a different
opportunity: constructing an architecture that can accurately simulate behavior across many
contexts. However, such an approach needs to avoid flattening agents into demographic
stereotypes, and measurement needs to advance beyond replication success or failure on average
treatment effects (16-19).
We present a generative agent architecture that simulates more than 1,000 real individuals using
two-hour qualitative interviews. The architecture combines these interviews with a large
language model to replicate individuals' attitudes and behaviors. By anchoring on individuals, we
can measure accuracy by comparing simulated attitudes and behaviors to the actual attitudes and
behaviors. We benchmark these agents using canonical social science measures such as the
General Social Survey (GSS; 20), the Big Five Personality Inventory (21), five well-known
behavioral economic games (e.g., the dictator game, a public goods game) (22-25), and five
social science experiments with control and treatment conditions that we sampled from a recent
large-scale replication effort (26-31). To support further research while protecting participant
privacy, we provide a two-pronged access system to the resulting agent bank: open access to
aggregated responses on fixed tasks for general research use, and restricted access to individual
responses on open tasks for researchers following a review process, ensuring the agents are
accessible while minimizing risks associated with the source interviews.
Creating 1,000 Generative Agents of Real People
To create simulations that better reflect the myriad, often idiosyncratic, factors that influence
individuals' attitudes, beliefs, and behaviors, we turn to in-depth interviews—a method that
previous work on predicting human life outcomes has employed to capture insights beyond what
can be obtained through traditional surveys and demographic instruments (32). In-depth
interviews, which combine pre-specified questions with adaptive follow-up questions based on
respondents' answers, are a foundational social science method with several advantages over
more structured data collection techniques (33, 34). While surveys with closed-ended questions
and predefined response categories are valuable for well-powered quantitative analysis and
hypothesis testing, semi-structured interviews offer distinct benefits for gaining idiographic
knowledge about individuals. Most notably, they give interviewees more freedom to highlight
what they find important, ultimately shaping what is measured.
2





Figure 1. The process of collecting participant data and creating generative agents begins by recruiting a stratified sample of 1,052
individuals from the U.S., selected based on age, census division, education, ethnicity, gender, income, neighborhood, political ideology,
and sexual identity. Once recruited, participants complete a two-hour audio interview with our AI interviewer, followed by surveys and
experiments. We create generative agents for each participant using their interview data. To evaluate these agents, both the generative
agents and participants complete the same surveys and experiments. For the human participants, this involves retaking the surveys and
experiments again two weeks later. We assess the accuracy of the agents by comparing agent responses to the participants' original
responses, normalizing by how consistently each participant successfully replicates their own responses two weeks later.
We recruited over 1,000 participants using stratified sampling to create a representative U.S.
sample across age, gender, race, region, education, and political ideology. Each participant
completed a voice-to-voice interview in English, producing transcripts with an average length of
6,491 words per participant (std = 2,541; SM 1). To facilitate this process, we developed an AI
interviewer (SM 2) that conducted the interview using a semi-structured interview protocol. To
avoid inadvertently tailoring the interview protocol to our evaluation metrics, we sought an
existing interview protocol that aimed for broad topical coverage. We selected an interview
protocol developed by sociologists as part of the American Voices Project (35). The script
explored a wide range of topics of interest to social scientists—from participants’ life stories
(e.g., “Tell me the story of your life—from your childhood, to education, to family and
relationships, and to any major life events you may have had”) to their views on current societal
issues (e.g., “How have you responded to the increased focus on race and/or racism and
policing?”; SM 8). Its broad scope, diverging from our metrics (e.g., while some questions
overlap thematically with the GSS, they do not directly include specific questions or cover
personality traits or economic game behaviors), strengthens results if high performance is
achieved. Within the interview’s structure and time limitations, the AI interviewer dynamically
generated follow-up questions tailored to each participant's responses.
3

![image p3_0](/mnt/data/images/p3_0.png)



To create the generative agents (14, 15), we developed a novel agent architecture that leverages
participants' full interview transcripts and a large language model (SM 3). When an agent is
queried, the entire interview transcript is injected into the model prompt, instructing the model to
imitate the person based on their interview data. For experiments requiring multiple
decision-making steps, agents were given memory of previous stimuli and their responses to
those stimuli through short text descriptions. The resulting agents can respond to any textual
stimulus, including forced-choice prompts, surveys, and multi-stage interactional settings.
We evaluated the generative agents on their ability to predict their source participants’ responses
to a series of surveys and experiments commonly used across social science disciplines. This
evaluation consisted of four components, which participants completed following their
interviews: the core module of the General Social Survey (GSS; 20), the 44-item Big Five
Inventory (BFI-44; 16), five well-known behavioral economic games (including the dictator
game, trust game, public goods game, and prisoner’s dilemma; 22-25), and five social science
experiments with control and treatment conditions (27-31). The experiments were sampled from
a recent large-scale replication effort (26), chosen based on criteria that the external replication
specified 1,000 participants for sufficient power and that the experiments could be delivered to
agents in text form (SM 4). We used the first three components to measure the accuracy of the
generative agents in predicting individual attitudes, traits, and behaviors, while the replication
studies assessed their ability to predict population-level treatment effects and effect sizes in a
well-powered replication. Our metrics and core analyses were pre-registered (SM 5).1
A key methodological benefit of simulating specific individuals is the ability to evaluate our
architecture by comparing how accurately each agent replicates the attitudes and behaviors of its
source individual. For the GSS, where responses are categorical, we measure accuracy and
correlation based on whether the agent selects the same survey response as the individual. For
the BFI-44 and economic games, which involve continuous responses, we assess accuracy and
correlation using mean absolute error (MAE). Since individuals often exhibit inconsistency in
their responses over time in both survey and behavioral studies (32, 36, 37), we use participants’
own attitudinal and behavioral consistency as a normalization factor: the probability of
accurately simulating an individual's attitudes or behaviors depends on how consistent those
attitudes and behaviors are over time.
To account for these varying levels in self-consistency, we asked each participant to complete our
battery twice, two weeks apart. Our main dependent variable is normalized accuracy, calculated
as the agent’s accuracy in predicting the individual’s responses divided by the individual’s own
replication accuracy. A normalized accuracy of 1.0 indicates that the generative agent predicts
the individual’s responses as accurately as the individual replicates their own responses two
weeks later. For continuous outcomes, we calculate normalized correlation instead.
1 Pre-registration materials: https://osf.io/mexkf/?view_only=375fe67b9a3e48afa7c3684c9d344da4
4





Figure 2. Generative agents’ predictive performance, and 95% confidence intervals. The consistency rate between participants and the
predictive performance of generative agents is evaluated across various constructs and averaged across individuals. For the General
Social Survey (GSS), accuracy is reported due to its categorical response types, while the Big Five personality traits and economic games
report mean absolute error (MAE) due to their numerical response types. Correlation is reported for all constructs. Normalized accuracy
is provided for all metrics, except for MAE, which cannot be calculated for individuals whose MAE is 0 (i.e., those who responded the
same way in both phases). We find that generative agents predict participants' behavior and attitudes well, especially when compared to
participants' own rate of internal consistency. Additionally, using interviews to inform agent behavior significantly improves the
predictive performance of agents for both GSS and Big Five constructs, outperforming other commonly used methods in the literature.
Predicting Individuals’ Attitudes and Behaviors
To assess the contribution of interviews to the generative agents' predictive accuracy, we
compared the performance of interview-based generative agents with two baselines that replace
interview transcripts with alternative forms of description. These baselines are grounded in how
language models have been used to proxy human behaviors in prior studies: one using
demographic attributes (13, 38), and the other using a paragraph summarizing the target person’s
profile (14). For the demographic-based generative agents, we used participants' responses to
GSS questions to capture individuals’ age, gender, race, and political ideology—demographic
attributes commonly used in previous studies (38). For the persona-based generative agents, we
asked participants to write a brief paragraph about themselves after the interview, including their
personal background, personality, and demographic details, similar to the material used to
generate persona agents in prior work (14).
The first component of our evaluation, the GSS, is widely used across sociology, political
science, social psychology, and other social sciences to assess respondents' demographic
backgrounds, behaviors, attitudes, and beliefs on a broad range of topics, including public policy,
race relations, gender roles, and religion (20). Our evaluation focused on 177 core GSS
5

![image p5_0](/mnt/data/images/p5_0.png)



questions, which we used to establish a benchmark for measuring the agents' predictive accuracy.
Each question had an average of 3.70 response options (std = 2.22), yielding a random chance
prediction accuracy of 27.03%.
For the GSS, the generative agents predicted participants’ responses with an average normalized
accuracy of 0.85 (std = 0.11), calculated from a raw accuracy of 68.85% (std = 6.01) divided by
participants’ replication accuracy of 81.25% (std = 8.11). These interview-based agents
significantly outperformed both demographic-based and persona-based agents (Figure 2), with a
margin of 14-15 normalized points. The demographic-based generative agents achieved a
normalized accuracy of 0.71 (std = 0.11), while persona-based agents reached 0.70 (std = 0.11).
An ANOVA of the accuracy rates rejected the null hypothesis of no significant difference (F(2,
3153) = 989.62, p < 0.001), and post-hoc pairwise Tukey tests confirmed that the
interview-based agents outperformed the other two groups.
The second component of our evaluation focused on predicting participants’ Big Five personality
traits using the BFI-44, which assesses five personality dimensions: openness, conscientiousness,
extraversion, agreeableness, and neuroticism (21). Each dimension is calculated as an aggregate
of eight to ten Likert scale questions. Our generative agents predicted participants' responses to
the individual items, which were then used to compute the predicted aggregate scores for each
personality dimension. These are continuous measures, so we calculated correlation coefficients
and normalized correlations.
For the Big Five, the generative agents achieved a normalized correlation of 0.80 (std = 1.88),
based on a raw correlation of r = 0.78 (std = 0.70) divided by participants’ replication correlation
of 0.95 (std = 0.76). As with the GSS, the interview-based generative agents outperformed both
demographic-based (normalized correlation = 0.55) and persona-based (normalized correlation =
0.75) agents. The interview-based agents also produced predictions with lower MAE for Big
Five personality traits (F(2, 3153) = 25.96, p < 0.001), and post-hoc pairwise Tukey tests
confirmed that interview-based agents significantly outperformed the other two groups.
The third component involved a series of five well-known economic games designed to elicit
participants' behaviors in decision-making contexts with real stakes. These included the Dictator
Game, the first and second player Trust Games, the Public Goods Game, and the Prisoner's
Dilemma (22-25). To ensure genuine engagement, participants were offered monetary incentives.
We standardized the output values for each game on a scale from 0 to 1 and compared the
generative agents' predicted values to the actual values obtained from participants. Since these
are continuous measures, we calculated correlation coefficients and normalized correlations. On
average, the generative agents achieved a normalized correlation of 0.66 (std = 2.83), derived
from a raw correlation of r = 0.66 (std = 0.95) divided by participants’ replication correlation of
0.99 (std = 1.00). However, there was no significant difference in MAE between the agents for
the economic games (F(2, 3153) = 0.12, p = 0.89).
In exploratory analyses, we tested the effectiveness and efficiency of interviews by comparing
interview-based generative agents to a baseline composite agent informed by participants' GSS,
Big Five, and economic game responses. We randomly sampled 100 participants and created
composite agents from their responses to these instruments. To prevent exact answer retrieval,
we excluded all question-answer pairs from the same category as the question being predicted
(categories were defined by the creators of each instrument), which excluded an average of
4.00% (std = 2.16). This composite agent serves as a baseline with access to semantically close
information to the evaluation, so any performance gap with the interview-based agents would
6





indicate the interview’s unique effectiveness in capturing participant identity. On average, the
composite generative agents achieved a normalized accuracy of 0.76 (std = 0.12) for the GSS, a
normalized correlation of 0.64 (std = 0.61) for the Big Five, and 0.31 (std = 1.22) for economic
games. These results still underperformed the interview-based generative agents.
We conducted additional tests by ablating portions of the generative agents' interviews to
examine the impact of interview content volume and style. First, even when we randomly
removed 80% of the interview transcript—equivalent to removing 96 minutes of the 120-minute
interview—the interview-based generative agents still outperformed the composite agents,
achieving an average normalized accuracy of 0.79 (std = 0.11) on the GSS, with similar results
observed for the Big Five. Second, to investigate whether the predictive power of interviews
stems from linguistic cues or the richness of the knowledge gained, we created
"interview-summary" generative agents by prompting GPT-4o to convert interview transcripts
into bullet-pointed summaries of key response pairs, capturing the factual content while
removing the original linguistic features. These agents also outperformed composite agents,
achieving a normalized accuracy of 0.83 (std = 0.12) on the GSS and showing similar
improvements for the Big Five. These findings suggest that, when informing language models
about human behavior, interviews are more effective and efficient than survey-based methods.
Table 1. Results of replication studies by human participants and generative agents. We report the p-values (***: < 0.001, **: <
0.01, *: < 0.05) and Cohen's d for effect sizes. Our replication with human participants replicated four out of five studies, while
generative agents informed by the interview transcript replicated the same four studies. The correlation of the effect sizes
between the human participants and generative agents achieved a strong correlation.
Predicting Experimental Replications
Participants took part in five social science experiments to assess whether generative agents can
predict treatment effects in experimental settings commonly used by social scientists. These were
drawn from a collection of published studies included in a large-scale replication effort (26-31;
7
Human replication
Agent prediction
Replication
Studies
Participants
Interview
Demog. Info.
Persona Desc.
p
Effect size
p
Effect size
p
Effect size
p
Effect size
Ames & Fiske
2015
***
9.45
***
12.59
***
13.43
***
10.03
Cooney et al.
2016
***
0.40
***
1.48
***
1.39
***
1.37
Halevy & Halali
2015
***
0.90
***
2.98
***
4.22
***
3.35
Rai et al.
2017
0.040
0.094
***
0.21
0.078
Schilke et al.
2015
***
0.33
***
2.97
***
5.52
***
3.74
Effect size
correlation
w/ human rep.
Correlation
Correlation
Correlation
r = 0.98
95% CI
[0.74, 0.99]
r = 0.93
95% CI
[0.24, 0.99]
r = 0.94
95% CI
[0.33, 0.99]





SM 4), including investigations of how perceived intent affects blame assignment (27) and how
fairness influences emotional responses (28). Both human participants in our work and
generative agents completed all five studies, with p-values and treatment effect sizes calculated
using the statistical methods as the original studies. Our participants successfully replicated the
results of four out of the five studies, failing to replicate one; the generative agents replicated the
same four studies and failed to replicate the fifth. The effect sizes estimated from the generative
agents were highly correlated with those of the participants (r = 0.98), compared to the
participants' internal consistency correlation of 0.99, yielding a normalized correlation of 0.99.
Figure 3. Demographic Parity Difference (DPD) for generative agents across political ideology, race, and gender subgroups on three
tasks: GSS (in percentages), Big Five, and economic games (in correlation coefficients). DPD represents the performance disparity
between the most and least favored groups within each demographic category. Generative agents using interviews consistently show
lower DPDs compared to those using demographic information or persona descriptions, suggesting that interview-based generative
agents mitigate bias more effectively across all tasks. Gender-based DPDs remain relatively low and consistent across all conditions.
Interviews Reduce Bias in Generative Agent Accuracy
There is concern about AI systems underperforming or misrepresenting underrepresented
populations (19). To address this concern, we conducted a subgroup analysis focusing on
political ideology, race, and gender—dimensions of particular interest in relevant literature (13,
38, 16-18). We aimed to assess whether the in-depth descriptions provided by interviews could
mitigate biases compared to methods using demographic prompts, which exhibited stereotyping
in prior research (16-19). We quantified bias using the Demographic Parity Difference (DPD),
which measures the difference in performance between the best performing and
worst-performing groups (39, 40). For the GSS, we report DPD in percentages; for Big Five and
8

![image p8_0](/mnt/data/images/p8_0.png)



economic games, in correlation coefficients. Subgroups were defined by participants' responses
to GSS items (details in SM 5).
Interview-based agents consistently reduced biases across tasks compared to demographic-based
agents. For political ideology, we observed that in the GSS, the DPD dropped from 12.35% for
demographic-based generative agents to 7.85% for interview-based generative agents. In the Big
Five personality traits, the DPD dropped from 0.165 to 0.063 (in correlation coefficients), and in
economic games, it dropped from 0.50 to 0.19 (in correlation coefficients). Although initial racial
subgroup discrepancies were smaller with demographic-based generative agents than the
interview-based generative agents, interview-based generative agents still reduced them further:
in the GSS, the DPD decreased from 3.33 to 2.08%; in the Big Five, from 0.17 to 0.11
correlation coefficients; and in economic games, from 0.043 to 0.040 correlation coefficients.
Gender-based DPD remained relatively constant across tasks, likely due to its already low level
of discrepancy.
Research Access for the Agent Bank
Access to an agent bank can help lay the foundations for replicable science using AI-based tools.
Our agent bank of 1,000 generative agents offers a resource toward these goals. To balance
scientific potential with privacy concerns, the authors at Stanford University provide a
two-pronged access system for research: open access to aggregated responses on fixed tasks
(e.g., GSS) and restricted access to individualized responses on open tasks. Safeguards include
usage audits, participant withdrawal options, and non-commercial use agreements, modeled after
genome banks and AI model deployments, supporting ethical research and reducing risk to
human subjects while enabling AI applications in the social sciences.2
Materials and Methods Summary
We contracted with the recruitment firm Bovitz (41) to obtain a U.S. sample of 1,000 individuals,
stratified by age, census division, education, ethnicity, gender, income, neighborhood, political
ideology, and sexual orientation. Participants completed interviews with the AI interviewer,
along with Qualtrics versions of the General Social Survey (GSS), Big Five personality
inventory, economic games, and selected experimental studies. For the GSS, we focused on 177
questions for the “core” module, excluding non-categorical questions, questions with more than
25 response options, and conditional questions. For the experimental studies, we selected five
studies from a recent large-scale replication effort (26-31). These were chosen based on two
inclusion criteria: first, the study had to be describable to a language model using text or images,
and second, the power analysis from the replication effort indicated that the effects would be
observable with 1,000 or fewer participants. This ensured that our human participants could
replicate the effects if present. The selected studies (27-31) covered the evaluation of harm based
on perceived intent, the role of fairness in emotional reactions, the perceived benefits of conflict
intervention, dehumanization in willingness to harm others, and how power influences trust.
2 The codebase for generating agent behavior is available as an open-source repository. Researchers interested in
constructing agents with their own data can access it here: https://github.com/joonspk-research/generative_agent
9





References
1. E. Bruch, J. Atwell, Agent-Based Models in Empirical Social Research. Sociological
Methods & Research 44, 186-221 (2015).
2. T. C. Schelling, Dynamic models of segregation. Journal of Mathematical Sociology 1,
143-186 (1971).
3. J. M. Epstein, R. L. Axtell, Growing Artificial Societies: Social Science from the Bottom
Up (The MIT Press, 1996).
4. R. Axtell, "Why agents? On the varied motivations for agent computing in the social
sciences" (Center on Social and Economic Dynamics Working Paper No. 17, 2000).
5. K. M. Carley, Organizational learning and personnel turnover. Organization Science 3,
20-46 (1992).
6. I. S. Lustick, PS-I: A user-friendly agent-based modeling platform for testing theories of
political identity and political stability. Journal of Artificial Societies and Social
Simulation 5, 3 (2002).
7. T. C. Schelling, Micromotives and Macrobehavior (W. W. Norton & Company, 1978).
8. E. Bonabeau, Agent-based modeling: Methods and techniques for simulating human
systems. Proc. Natl. Acad. Sci. U.S.A. 99 (suppl. 3), 7280-7287 (2002);
https://doi.org/10.1073/pnas.082080899
9. M. W. Macy, R. Willer, From Factors to Actors: Computational Sociology and
Agent-Based Modeling. Annu. Rev. Sociol. 28, 143-166 (2002);
https://doi.org/10.1146/annurev.soc.28.110601.141117
10. J. von Neumann, O. Morgenstern, Theory of Games and Economic Behavior (Princeton
University Press, 1944).
11. McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P.
Zarembka (Ed.), Frontiers in Econometrics (pp. 105-142). Academic Press.
12. J. J. Horton, "Large language models as simulated economic agents: What can we learn
from homo silicus?" (2023).
13. A. Ashokkumar, L. Hewitt, I. Ghezae, R. Willer, "Predicting Results of Social Science
Experiments Using Large Language Models" (2024).
14. J. S. Park, L. Popowski, C. J. Cai, M. R. Morris, P. Liang, M. S. Bernstein, Social
simulacra: Creating Populated Prototypes for Social Computing Systems, in Proceedings
of the 35th Annual ACM Symposium on User Interface Software and Technology (ACM,
2022).
15. J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, M. S. Bernstein, Generative
agents: Interactive simulacra of human behavior, in Proceedings of the 36th Annual ACM
Symposium on User Interface Software and Technology (ACM, 2023).
16. M. Cheng, T. Piccardi, D. Yang, in Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing (EMNLP 2023) (Association for
Computational Linguistics, 2023).
17. A. Wang, J. Morgenstern, J. P. Dickerson, "Large language models cannot replace human
participants because they cannot portray identity groups" (2024).
18. S. Santurkar, E. Durmus, F. Ladhak, C. Lee, P. Liang, T. Hashimoto, in Proceedings of
the 40th International Conference on Machine Learning (ICML '23) (PMLR, 2023).
19. L. Messeri, M. J. Crockett, Artificial intelligence and illusions of understanding in
scientific research. Nature 627, 49-58 (2024).
20. National Opinion Research Center, "General Social Survey, 2023" (NORC at the
University of Chicago, 2023); https://gss.norc.org.
10





21. O. P. John, S. Srivastava, The Big Five trait taxonomy: History, measurement, and
theoretical perspectives, in Handbook of Personality: Theory and Research, L. A. Pervin,
O. P. John, Eds. (Guilford Press, ed. 2, 1999), pp. 102-138.
22. R. Forsythe, J. L. Horowitz, N. E. Savin, M. Sefton, Fairness in simple bargaining
experiments. Games and Economic Behavior 6, 347-369 (1994).
23. J. Berg, J. Dickhaut, K. McCabe, Trust, reciprocity, and social history. Games and
Economic Behavior 10, 122-142 (1995).
24. J. O. Ledyard, in The Handbook of Experimental Economics, J. H. Kagel, A. E. Roth,
Eds. (Princeton University Press, 1995), pp. 111-194.
25. A. Rapoport, A. M. Chammah, Prisoner's Dilemma: A Study in Conflict and Cooperation
(University of Michigan Press, 1965).
26. C. F. Camerer et al., "Mechanical Turk Replication Project" (2024);
https://mtrp.info/index.html.
27. D. L. Ames, S. T. Fiske, Perceived intent motivates people to magnify observed harms.
PNAS 112, 3599-3605 (2015).
28. G. Cooney, D. T. Gilbert, T. D. Wilson, When fairness matters less than we expect. PNAS
113, 11168-11171 (2016).
29. N. Halevy, E. Halali, Selfish third parties act as peacemakers by transforming conflicts
and promoting cooperation. PNAS 112, 6937-6942 (2015).
30. T. S. Rai, P. Valdesolo, J. Graham, Dehumanization increases instrumental violence, but
not moral violence. PNAS 114, 8511-8516 (2017).
31. O. Schilke, M. Reimann, K. S. Cook, Power decreases trust in social exchange. PNAS
112, 12950-12955 (2015).
32. I. Lundberg et al., The origins of unpredictability in life outcome prediction tasks. Proc.
Natl. Acad. Sci. U.S.A. 121, e2322973121 (2024).
33. A. Lareau, Listening to People: A Practical Guide to Interviewing, Participant
Observation, Data Analysis, and Writing It All Up (Univ. of Chicago Press, 2021).
34. R. S. Weiss, Learning From Strangers: The Art and Method of Qualitative Interview
Studies (Free Press, 1994).
35. Stanford Center on Poverty and Inequality, "American Voices Project" (2021);
https://inequality.stanford.edu/avp/methodology.
36. S. Ansolabehere, J. Rodden, J. M. Snyder Jr., The Strength of Issues: Using Multiple
Measures to Gauge Preference Stability, Ideological Constraint, and Issue Voting.
American Political Science Review 102, 215-232 (2008).
37. M. J. Salganik et al., Measuring the predictability of life outcomes with a scientific mass
collaboration. Proc. Natl. Acad. Sci. U.S.A. 117, 8398-8403 (2020).
38. L. P. Argyle et al., Out of one, many: Using language models to simulate human samples.
Political Analysis 31, 337-355 (2023).
39. M. Hardt, E. Price, N. Srebro, Equality of opportunity in supervised learning, in
Advances in Neural Information Processing Systems 29 (2016), pp. 3315-3323;
arXiv:1610.02413.
40. S. Barocas, M. Hardt, A. Narayanan, Fairness and Machine Learning (2019);
https://fairmlbook.org
41. M. N. Stagnaro, J. Druckman, A. J. Berinsky, A. A. Arechar, R. Willer, D. Rand,
Representativeness versus Response Quality: Assessing Nine Opt-In Online Survey
Samples. OSF Preprints [Preprint] (2024); https://osf.io/preprints/psyarxiv/h9j2dc
42. T. W. Smith, M. Davern, J. Freese, S.L. Morgan, "General Social Surveys, 1972-2020:
Cumulative Codebook" (NORC at the University of Chicago, 2021);
11





https://gss.norc.org/content/dam/gss/get-documentation/pdf/other/2020%20GSS%20Repl
icating%20Core.pdf
43. R. M. Groves, F. J. Fowler Jr., M. P. Couper, J. M. Lepkowski, E. Singer, R. Tourangeau,
Survey Methodology (John Wiley & Sons, ed. 2, 2009).
44. S. Brinkmann, S. Kvale, InterViews: Learning the Craft of Qualitative Research
Interviewing (SAGE Publications, ed. 3, 2014).
45. N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, P. Liang, Lost in the
middle: How language models use long contexts. Transactions of the Association for
Computational Linguistics 11, 1312-1327 (2024).
46. B. Reeves, C. Nass, The Media Equation: How People Treat Computers, Television, and
New Media Like Real People and Places (Cambridge University Press, 1996).
47. OpenAI, "Text to speech guide" (2024);
https://platform.openai.com/docs/guides/text-to-speech (accessed 20 September 2024).
48. OpenAI, "Whisper" (2024); https://openai.com/research/whisper (accessed 20 September
2024).
49. OpenAI, "GPT-4" (2024); https://openai.com/research/gpt-4 (accessed 20 September
2024).
50. P. V. Marsden, T. W. Smith, M. Hout, Tracking US Social Change Over a Half-Century:
The General Social Survey at Fifty. Annual Review of Sociology 46, 109-134 (2020).
51. NORC at the University of Chicago, "General Social Surveys, 1972-2022: Cumulative
Codebook" (2023);
https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202022%20
Codebook.pdf (accessed 20 September 2024).
52. S. C. Schmitt, J. J. Gaughan, B. N. Doritya, A. L. Gonzalez, L. D. Smillie, R. E. Lucas,
D. B. Nelson, M. Brent Donnellan, The Big Five Across Time, Space, and Method: A
Systematic Review. PsyArXiv [Preprint] (2023); https://doi.org/10.31234/osf.io/37w8p
53. Y. Strus, J. Cieciuch, Toward a synthesis of personality, temperament, motivation,
emotion and mental health models within the Circumplex of Personality Metatraits.
Journal of Research in Personality 82, 103844 (2019).
54. C. F. Camerer, Behavioral Game Theory: Experiments in Strategic Interaction (Princeton
University Press, 2003).
55. B. A. Nosek, G. Alter, G. C. Banks, D. Borsboom, S. D. Bowman, S. J. Breckler, S.
Buck, C. D. Chambers, G. Chin, G. Christensen, M. Contestabile, A. Dafoe, E. Eich, J.
Freese, R. Glennerster, D. Goroff, D. P. Green, B. Hesse, M. Humphreys, J. Ishiyama, D.
Karlan, A. Kraut, A. Lupia, P. Mabry, T. Madon, N. Malhotra, E. Mayo-Wilson, M.
McNutt, E. Miguel, E. Levy Paluck, U. Simonsohn, C. Soderberg, B. A. Spellman, J.
Turitto, G. VandenBos, S. Vazire, E. J. Wagenmakers, R. Wilson, T. Yarkoni, Promoting
an open research culture. Science 348, 1422-1425 (2015).
56. J. Chandler, M. Cumpston, T. Li, M. J. Page, V. J. H. W. Welch, Cochrane Handbook for
Systematic Reviews of Interventions (Wiley, ed. 2, 2019).
57. Silver, N. C., & Dunlap, W. P. (1987). Averaging correlation coefficients: Should Fisher's
z transformation be used? Journal of Applied Psychology, 72(1), 146-148.
12



