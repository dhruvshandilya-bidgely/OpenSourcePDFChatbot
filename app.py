import streamlit as st
import streamlit.components.v1 as stc
from langchain_community.document_loaders import PDFPlumberLoader
import time

from backend.rag import rag_bot, create_vectorstore
from backend.select_plus_map_red import final_step

from utils.load_text_from_pdf import text_from_pdf_loader
from utils.docs_from_text_string import document_from_text

if "proc_button_rag" not in st.session_state:
	st.session_state["proc_button_rag"] = False

def set_proc_button_rag():
	st.session_state.proc_button_rag = True

if "vect_button_rag" not in st.session_state:
	st.session_state["vect_button_rag"] = False

def set_vect_button_rag():
	st.session_state.vect_button_rag = True

if "proc_button_sel" not in st.session_state:
	st.session_state["proc_button_sel"] = False

def set_proc_button_sel():
	st.session_state.proc_button_sel = True

def main():
	st.title("Open Source chatbot")
	
	menu = ["RAG","Selection plus map reduce"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "RAG":
		if "proc_button_rag" not in st.session_state:
			st.session_state["proc_button_rag"] = None
		
		st.subheader("Select pdf file to be uploaded")
		docx_file = st.file_uploader("Upload File",type=['pdf'])
		st.button("Start processing pdf", on_click=set_proc_button_rag)
		if st.session_state.proc_button_rag:
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)

				with open(docx_file.name, mode='wb') as w:
					w.write(docx_file.getvalue())

				# Check File Type
				if docx_file.type == "application/pdf":

					loader = PDFPlumberLoader(docx_file.name)
					first_page = loader.load()[0].page_content
					with st.expander("Show first page content"):
						st.text(first_page)
							
			st.button("Vectorise the pdf", on_click=set_vect_button_rag)
			if st.session_state.vect_button_rag:
				with st.spinner(f'Creating vector embedding for {docx_file.name}'):
					vector = create_vectorstore(docx_file.name)
				st.success(f'Vector embedding creation done!')

				# Initialize chat history
				if "messages" not in st.session_state:
					st.session_state.messages = []

				# Display chat messages from history on app rerun
				for message in st.session_state.messages:
					with st.chat_message(message["role"]):
						st.markdown(message["content"])

				# React to user input
				if prompt := st.chat_input("Enter your question here."):
					# Display user message in chat message container
					st.chat_message("user").markdown(prompt)
					# Add user message to chat history
					st.session_state.messages.append({"role": "user", "content": prompt})
					with st.spinner('Generating response'):
						start_time = time.time()
						response = rag_bot(vectorstore=vector, question=prompt)
						response_processed = response.replace('$', '\$')
						time_str = f"\n\n--- {(time.time() - start_time)} seconds for response ---"
						response_processed += time_str
						print(response)
					# Display assistant response in chat message container
					with st.chat_message("assistant"):
						st.markdown(response_processed)
					# Add assistant response to chat history
					st.session_state.messages.append({"role": "assistant", "content": response_processed})

	elif choice == "Selection plus map reduce":

		st.subheader("Select pdf file to be uploaded")
		docx_file = st.file_uploader("Upload File",type=['pdf'])
		st.button("Start processing pdf", on_click=set_proc_button_sel)
		if st.session_state.proc_button_sel:
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)

				with open(docx_file.name, mode='wb') as w:
					w.write(docx_file.getvalue())

				# Check File Type
				if docx_file.type == "application/pdf":

					loader = PDFPlumberLoader(docx_file.name)
					first_page = loader.load()[0].page_content
					with st.expander("Show first page content"):
						st.text(first_page)
				
				txt = text_from_pdf_loader(pdf_location=docx_file.name)
				docs = document_from_text(txt)

				# Initialize chat history
				if "messages2" not in st.session_state:
					st.session_state.messages2 = []

				# Display chat messages from history on app rerun
				for message in st.session_state.messages2:
					with st.chat_message(message["role"]):
						st.markdown(message["content"])

				# React to user input
				if prompt := st.chat_input("Enter your question here."):
					# Display user message in chat message container
					st.chat_message("user").markdown(prompt)
					# Add user message to chat history
					st.session_state.messages2.append({"role": "user", "content": prompt})
					with st.spinner('Generating response'):
						start_time = time.time()
						response = final_step(docs=docs, question=prompt)
						response_processed = response.replace('$', '\$')
						time_str = f"\n\n--- {(time.time() - start_time)} seconds for response ---"
						response_processed += time_str
						print(response)
					# Display assistant response in chat message container
					with st.chat_message("assistant"):
						st.markdown(response_processed)
					# Add assistant response to chat history
					st.session_state.messages2.append({"role": "assistant", "content": response_processed})


					
if __name__ == '__main__':
	main()