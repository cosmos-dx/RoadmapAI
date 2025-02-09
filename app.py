from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from crewai import Agent, Task, Crew, LLM
import json
import os
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

load_dotenv()

app = Flask(__name__)

llm = LLM(model="gemini/gemini-2.0-flash-exp", temperature=0.7)


class RoadmapGenerator:
    """Handles roadmap generation based on user input."""
    def __init__(self, llm):
        self.llm = llm

        # Define Agents
        self.roadmap_agent = Agent(
            name="Roadmap Creator",
            role="Tech Roadmap Designer",
            goal="Create a personalized learning roadmap based on extracted user knowledge.",
            backstory="An expert at structuring learning paths for any tech topic, considering prior knowledge.",
            verbose=True,
            llm=self.llm,
        )

        self.validator_agent = Agent(
            name="Roadmap Validator",
            role="Quality Assurance Specialist",
            goal="Validate and enhance the roadmap to ensure completeness and correctness.",
            backstory="A detail-oriented specialist refining learning roadmaps.",
            verbose=True,
            llm=self.llm,
        )

        # Define Tasks
        self.research_task = Task(
            description="Analyze user's current knowledge of {tech_topic} and suggest next learning steps.",
            expected_output="A knowledge assessment highlighting gaps and recommended next steps.",
            agent=self.roadmap_agent
        )

        self.roadmap_task = Task(
            description="Create a structured learning roadmap for {tech_topic} based on user's knowledge of {current_skills}.",
            expected_output="A step-by-step roadmap with milestones, skills, and learning resources customized to the user.",
            agent=self.roadmap_agent,
            depends_on=[self.research_task]
        )

        self.validation_task = Task(
            description="Review and refine the roadmap for {tech_topic} to ensure it's structured, practical, and useful.",
            expected_output="A high-quality roadmap with verified content.",
            agent=self.validator_agent,
            depends_on=[self.roadmap_task]
        )

        # Define Crew
        self.crew = Crew(
            agents=[self.roadmap_agent, self.validator_agent],
            tasks=[self.research_task, self.roadmap_task, self.validation_task]
        )

    def generate_roadmap(self, tech_topic, current_skills):
        """Runs the roadmap generation process."""
        result = self.crew.kickoff(inputs={"tech_topic": tech_topic, "current_skills": current_skills})
        return result


class CareerAdvisory:
    """Handles career guidance based on the chosen tech topic."""

    def __init__(self, llm):
        self.llm = llm

        # Define Career Agent
        self.career_coach_agent = Agent(
            name="Career Coach",
            role="Career Advisor",
            goal="Align the learning roadmap with career opportunities.",
            backstory="An industry expert who maps learning paths to job roles.",
            verbose=True,
            llm=self.llm,
        )

        # Define Task
        self.career_task = Task(
            description="Identify career opportunities related to {tech_topic} and suggest necessary skills, certifications, and job roles.",
            expected_output="A list of career paths, relevant job roles, and required skills for {tech_topic}.",
            agent=self.career_coach_agent
        )

        # Define Crew
        self.crew = Crew(
            agents=[self.career_coach_agent],
            tasks=[self.career_task]
        )

    def get_career_guidance(self, tech_topic):
        """Runs the career advisory process."""
        result = self.crew.kickoff(inputs={"tech_topic": tech_topic})
        return result


class ProjectGuide:
    """Suggests hands-on projects for different learning levels."""

    def __init__(self, llm):
        self.llm = llm

        # Define Project Agent
        self.project_guide_agent = Agent(
            name="Project Guide",
            role="Hands-on Project Consultant",
            goal="Suggest projects to apply knowledge of {tech_topic} at different levels.",
            backstory="A mentor focused on practical learning with real-world projects.",
            verbose=True,
            llm=self.llm,
        )

        # Define Task
        self.project_task = Task(
            description="Suggest hands-on projects for {tech_topic} at beginner, intermediate, and advanced levels.",
            expected_output="A structured list of projects with tools, frameworks, and GitHub links.",
            agent=self.project_guide_agent
        )

        # Define Crew
        self.crew = Crew(
            agents=[self.project_guide_agent],
            tasks=[self.project_task]
        )

    def get_project_recommendations(self, tech_topic):
        """Runs the project recommendation process."""
        result = self.crew.kickoff(inputs={"tech_topic": tech_topic})
        return result


class TechRoadmapSystem:
    """Main system that integrates Roadmap, Career, and Project Guidance."""
    def __init__(self):
        self.llm = LLM(model="gemini/gemini-2.0-flash-exp", temperature=0.7)

        self.roadmap_generator = RoadmapGenerator(self.llm)
        self.career_advisory = CareerAdvisory(self.llm)
        self.project_guide = ProjectGuide(self.llm)

    def extract_user_input(self, user_input):
        """Extracts the tech topic and user's skills using an LLM chain."""
        extraction_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""
            Extract the tech topic and the user's existing knowledge from the following input:
            "{user_input}"

            Respond in JSON format:
            {{"tech_topic": "...", "current_skills": "..."}}
            """
        )

        extraction_chain = LLMChain(prompt=extraction_prompt, llm=llm)
        unprocessed_extracted_data = extraction_chain.run(user_input)
        unprocessed_extracted_data = str(unprocessed_extracted_data)
        unprocessed_extracted_data = unprocessed_extracted_data.replace("```", '')
        unprocessed_extracted_data = unprocessed_extracted_data.replace("JSON", '')
        unprocessed_extracted_data = unprocessed_extracted_data.replace("json", '')
        print(unprocessed_extracted_data)
        extracted_data = json.loads(unprocessed_extracted_data)

        return extracted_data["tech_topic"], extracted_data["current_skills"]

    def generate_full_roadmap(self, user_input):
        """Executes the full roadmap generation process."""
        tech_topic, current_skills = self.extract_user_input(user_input)

        roadmap = self.roadmap_generator.generate_roadmap(tech_topic, current_skills)
        career_guidance = self.career_advisory.get_career_guidance(tech_topic)
        projects = self.project_guide.get_project_recommendations(tech_topic)

        full_roadmap = f"""
        # Personalized Roadmap for {tech_topic}

        ## 📌 Learning Roadmap
        {roadmap}

        ## 💼 Career Opportunities
        {career_guidance}

        ## 🛠️ Suggested Projects
        {projects}
        """

        self.save_as_markdown(tech_topic, full_roadmap)
        return full_roadmap

    def save_as_markdown(self, tech_topic, content):
        """Saves the generated roadmap as a markdown file."""
        md_filename = f"roadmap_{tech_topic.replace(' ', '_')}.md"

        with open(md_filename, "w", encoding="utf-8") as md_file:
            md_file.write(content)

        print(f"Roadmap saved as {md_filename}")
    def generate_basic_roadmap(self, user_input):
        """Executes the basic roadmap generation process."""
        tech_topic, result_or_message = self.extract_user_input(user_input)
        
        if tech_topic is None:
            return result_or_message  # Return error message

        roadmap = self.roadmap_generator.generate_roadmap(tech_topic, result_or_message)

        basic_roadmap = f"""
        # Personalized Roadmap for {tech_topic}

        ## 📌 Learning Roadmap
        {roadmap}
        """

        self.save_as_markdown(tech_topic, basic_roadmap)
        return basic_roadmap


# # Example Usage
# if __name__ == "__main__":
#     system = TechRoadmapSystem()
#     user_input = "I want to know the roadmap of Golang"
#     # user_input1 = "I want to know the roadmap of Redis."
#     # roadmap_output = system.generate_full_roadmap(user_input)
#     roadmap_basic = system.generate_basic_roadmap(user_input)
#     print("\n🚀 Generated Roadmap:\n")
#     print(roadmap_basic)


@app.route('/generate-roadmap', methods=['GET'])
def generate_roadmap():
    """Endpoint to generate a tech roadmap based on query and type."""
    user_input = request.args.get('query')
    roadmap_type = request.args.get('type')

    if not user_input or not roadmap_type:
        return jsonify({"error": "Both 'query' and 'type' parameters are required"}), 400
    system = TechRoadmapSystem()
    try:
        if roadmap_type == "full":
            roadmap = system.generate_full_roadmap(user_input)
        elif roadmap_type == "basic":
            roadmap = system.generate_basic_roadmap(user_input)
        else:
            return jsonify({"error": "Invalid type. Use 'full' or 'basic'."}), 400

        return jsonify({"roadmap": roadmap})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # app.run(host=os.getenv('HOST'), port=int(os.getenv('PORT'),8080), debug=False)
    app.run(
        host=os.getenv('HOST', '0.0.0.0'), 
        port=int(os.getenv('PORT', '8080'))
        debug=False
    )
