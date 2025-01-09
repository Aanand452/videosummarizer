import streamlit as st
from phi.agent import Agent 
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai
import time 
from pathlib import Path 

from dotenv import load_dotenv
load_dotenv()
import os 

import tempfile


API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

## page configuration
st.set_page_config(
    page_title="Multimodel AI AGENT - Video Summarizer",
    page_icon=":video_camera:",
    layout="wide"
)


st.title("Phidata Video AI Summarizer AI Agent :video_camera: ")
st.header("Powered by gemini 2.0 Flash Exp")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools = [DuckDuckGo()],
        markdown=True,
    )


## Initialize the agent
multimodal_Agent = initialize_agent()

## File uploader 
video_file = st.file_uploader(
    "Upload a file",type=["mp4","mov","avi"],help="Upload a video for AI analyst"
)


if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format='video/mp4', start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from this video?",
        placeholder="Ask Anything about the video content",
        help="Provide specific questions or insights you want from the video"
    )

    if(st.button("Analyze Video", key="analyze_video_button")):
        if not user_query:
            st.warning("Please provide a query to analyze the video")
        else:
            try:
                
                with st.spinner("Pocessing Video and gathering insights..."):
                    #Upload and process the video file
                    process_video = upload_file(video_path)
                    while process_video.state.name == "PROCESSING":
                        time.sleep(2)
                        process_video = get_file(process_video.name)

                    #Prompt generation for analysis
                    analysis_prompt = (
                        f"""Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary we research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.

                        """
                    )

                    #AI agent processing
                    response = multimodal_Agent.run(analysis_prompt, videos=[process_video])

                #Display the response
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                #Clean up the temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis")

#Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea{
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

