# txt = text_from_pdf_loader(pdf_location='Jungle.pdf')
# txt = txt.split('The White Seal', 1)[0]

from backend.select_plus_map_red import final_step
from utils.docs_from_text_string import document_from_text
from utils.load_text_from_pdf import text_from_pdf_loader
import time

txt = text_from_pdf_loader(pdf_location='sample_pdfs/the_gift_of_the_magi.pdf')

docs = document_from_text(txt)

# for i in range(len(docs)):
#     f = open(f"docs/{i}.txt", "w")
#     f.write(docs[i].page_content)
#     f.close()

question = "What gift had Jim brought for Della?"

start_time = time.time()
print('Started execution.')

ans_str = final_step(docs,question)

print(ans_str)
print("\n--- %s seconds ---" % (time.time() - start_time))