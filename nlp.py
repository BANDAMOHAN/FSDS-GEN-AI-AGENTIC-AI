import streamlit as st
import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import bigrams,trigrams,ngrams
from nltk import PorterStemmer
from nltk import pos_tag
from wordcloud import WordCloud
from nltk.tokenize import blankline_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

st.title("NLP Introduction")
st.subheader("Here's the detailed about words and sentences that is extracted from paragraph")
st.text("The Internet is one of the most transformative inventions in human history, connecting billions of people across the globe and shaping the way societies function in the 21st century. Originally developed in the late 20th century as a research project to share information between universities and government institutions, it has grown into a vast, decentralized network that powers communication, commerce, education, and entertainment. Through the Internet, people can instantly exchange ideas, access knowledge that was once locked away in libraries, and collaborate across continents in real time. It has given rise to new industries, from e-commerce and cloud computing to social media and artificial intelligence, while also changing traditional fields such as banking, healthcare, and education. Despite its incredible benefits, the Internet also presents challenges such as misinformation, privacy concerns, and cybercrime, reminding us that every powerful tool requires responsible use. As technology continues to evolve with the expansion of high-speed networks, the Internet of Things, and future quantum communication systems, the Internet will likely become even more deeply integrated into our daily lives, shaping how future generations learn, work, and connect with the world.")

paragraph ="The Internet is one of the most transformative inventions in human history, connecting billions of people across the globe and shaping the way societies function in the 21st century. Originally developed in the late 20th century as a research project to share information between universities and government institutions, it has grown into a vast, decentralized network that powers communication, commerce, education, and entertainment. Through the Internet, people can instantly exchange ideas, access knowledge that was once locked away in libraries, and collaborate across continents in real time. It has given rise to new industries, from e-commerce and cloud computing to social media and artificial intelligence, while also changing traditional fields such as banking, healthcare, and education. Despite its incredible benefits, the Internet also presents challenges such as misinformation, privacy concerns, and cybercrime, reminding us that every powerful tool requires responsible use. As technology continues to evolve with the expansion of high-speed networks, the Internet of Things, and future quantum communication systems, the Internet will likely become even more deeply integrated into our daily lives, shaping how future generations learn, work, and connect with the world."

words=word_tokenize(paragraph)
st.button('Click:',words)
A=words
sents=sent_tokenize(paragraph)
B=sents
bi=list(nltk.bigrams(paragraph))

tri=list(nltk.trigrams(paragraph))

ngram=list(nltk.ngrams(paragraph,4))

blank=blankline_tokenize(paragraph)

pst=PorterStemmer()

stm=pst.stem(paragraph)



pst_tags=nltk.pos_tag(words)

wordcloud=WordCloud(width=480,height=480,margin=0).generate(paragraph)

picture=plt.imshow(wordcloud,interpolation='bilinear'),plt.axis("off"),plt.margins(x=0,y=0),plt.show()


st.write("If you want to words in the above paragraph.Click the below button")
st.button('click')

st.write("If you want to sentences in the above paragraph.Click the below button")
st.button('sents')
st.radio('Pick One',['A','B'])