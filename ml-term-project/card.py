import streamlit as st

gDB = [["img1","title1","contents1"],
       ["img2","title2","contents2"],
       ["img3","title3","contents3"],
       ["img4","title4","contents4"],
       ["img5","title5","contents5"]]

st.header("Let's try a simple test.")
st.divider()

dislikeGame = []
likeGame = []

# 세션 상태 초기화
if 'g_index' not in st.session_state:
    st.session_state.g_index = 0

with st.container():
    # 카드

    dislike, game, like = st.columns([1,3,1])

    with dislike:
        st.empty()
        dislikeBtn = st.button("DISLIKE")
        
    with game:
        testGameInfo = st.container(border=True)
        testGameInfo.write(gDB[st.session_state.g_index][0])
        testGameInfo.subheader(gDB[st.session_state.g_index][1])
        testGameInfo.write(gDB[st.session_state.g_index][2])
        
    with like:
        st.empty()
        likeBtn = st.button("LIKE")

if dislikeBtn:
    dislikeGame.append(gDB[st.session_state.g_index])
    st.session_state.g_index += 1

if likeBtn:
    likeGame.append(gDB[st.session_state.g_index])
    st.session_state.g_index += 1
    
