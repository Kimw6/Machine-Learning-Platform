import streamlit as st
import tutorial_assets as ta

st.subheader("`Welcome to Tutorials and Resources`")

def render_tutorials(name):   
    container = st.container()
    with container.expander(f"### `{name}`"):
        render_hyperlinks(ta.models[name]['hyperlinks'])
        render_video_links(ta.models[name]['video_links'])



def render_video_links(video_links):
    st.subheader("`Video Tutorials`")
    for link in video_links:
        st.markdown(f"{link['title']}")
        st.write(f"Link to Video: {link['url']}")
        st.video(link['url'])
        


def render_hyperlinks(hyperlinks):
    st.subheader("`Links to Resources`")
    for link in hyperlinks:
        st.write(f"[{link['title']}]({link['url']})")





def render_resouces():
    for key, value in ta.models.items():
        if value is not None:
            render_tutorials(key)


render_resouces()

