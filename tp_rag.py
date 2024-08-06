from backend.rag import create_vectorstore, rag_bot
import time

file_loc = "sample_pdfs/the_gift_of_the_magi.pdf"

start_time = time.time()
print('Started Vectorization.')

vector = create_vectorstore(file_loc)

print("\n--- %s seconds for vectorisation ---" % (time.time() - start_time))

start_time = time.time()
print('Started Answering.')

question = "What gift had Jim brought for Della?"
ans = rag_bot(vector, question)
print(ans)

print("\n--- %s seconds for answering ---" % (time.time() - start_time))