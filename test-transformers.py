from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Yangi versiyaga mos tarzda model va tokenizatorni yuklaymiz
model_id = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

text = """
Transformerlar - bu tabiiy tilni qayta ishlash (NLP) vazifalarini bajarish uchun ishlatiladigan chuqur o'rganish modellarining bir turi. Ular o'zlarining o'ziga xos arxitekturasi bilan mashhur bo'lib, bu arxitektura parallel ishlov berish va uzoq muddatli bog'lanishlarni samarali tarzda ushlab turishga imkon beradi. Transformerlardan matnni tarjima qilish, matnni yaratish, savol-javob tizimlari va boshqa ko'plab NLP vazifalarida foydalaniladi. Ular katta hajmdagi ma'lumotlar bilan o'qitilgan va kontekstni tushunishda yuqori aniqlikka ega. Transformerlardan foydalanish orqali tabiiy tilni qayta ishlash sohasida sezilarli yutuqlarga erishildi.
Transformerlar o'zlarining "self-attention" mexanizmi bilan ajralib turadi, bu esa modelga kirish matnidagi har bir so'zning boshqa so'zlar bilan qanday bog'liqligini anixlash imkonini beradi. Bu mexanizm modelga kontekstni yaxshiroq tushunishga yordam beradi va natijada yanada aniq va mos javoblar beradi. Transformerlar, shuningdek, parallel ishlov berish imkoniyatiga ega bo'lib, bu ularni katta hajmdagi ma'lumotlar bilan ishlashda samarali qiladi. Shu sababli, ular ko'plab zamonaviy NLP tizimlarining asosiy komponenti hisoblanadi.
"""

# Matnni model tushunadigan formatga o'tkazamiz
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# Konspekt (summary) hosil qilamiz
summary_ids = model.generate(inputs["input_ids"], max_length=100, min_length=30, do_sample=False)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)







# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # 1. Dunyodagi eng mashhur qisqartirish modelini va uning tokenizerini yuklaymiz
# model_name = "facebook/bart-large-cnn"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# text = """
# Transformerlar - bu tabiiy tilni qayta ishlash (NLP) vazifalarini bajarish uchun ishlatiladigan chuqur o'rganish modellarining bir turi. Ular o'zlarining o'ziga xos arxitekturasi bilan mashhur bo'lib, bu arxitektura parallel ishlov berish va uzoq muddatli bog'lanishlarni samarali tarzda ushlab turishga imkon beradi. Transformerlardan matnni tarjima qilish, matnni yaratish, savol-javob tizimlari va boshqa ko'plab NLP vazifalarida foydalaniladi. Ular katta hajmdagi ma'lumotlar bilan o'qitilgan va kontekstni tushunishda yuqori aniqlikka ega. Transformerlardan foydalanish orqali tabiiy tilni qayta ishlash sohasida sezilarli yutuqlarga erishildi.
# Transformerlar o'zlarining "self-attention" mexanizmi bilan ajralib turadi, bu esa modelga kirish matnidagi har bir so'zning boshqa so'zlar bilan qanday bog'liqligini aniqlash imkonini beradi. Bu mexanizm modelga kontekstni yaxshiroq tushunishga yordam beradi va natijada yanada aniq va mos javoblar beradi. Transformerlar, shuningdek, parallel ishlov berish imkoniyatiga ega bo'lib, bu ularni katta hajmdagi ma'lumotlar bilan ishlashda samarali qiladi. Shu sababli, ular ko'plab zamonaviy NLP tizimlarining asosiy komponenti hisoblanadi.
# """

# # 2. Matnni model tushunadigan raqamlarga (tokenlarga) o'giramiz
# inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)

# # 3. Modelga qisqartirish buyrug'ini beramiz
# summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=100, min_length=30, early_stopping=True)

# # 4. Raqamlarni qaytadan odam tushunadigan matnga o'giramiz
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print("\n--- QISQARTIRILGAN MATN ---")
# print(summary)