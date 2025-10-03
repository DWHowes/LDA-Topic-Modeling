import re

class process():
    def __init__(self) -> None:
        pass

    def tokenize(self, doc)->str:
        lst = []
        lst = [token.text for token in doc]
        return " ".join(lst)
    
    def lower(self, doc)->str:
        lst = []
        lst = [token.text.lower() for token in doc]
        return " ".join(lst)
    
    def del_punct(self, doc)->str:
        lst = []
        lst = [token.text for token in doc if not token.is_punct]
        return " ".join(lst)

    def del_stop(self, doc)->str:
        lst = []
        lst = [token.text for token in doc if not token.is_stop]
        return " ".join(lst)
    
    def remove_ws(self, text:str)->str:
        text = text.strip()
        return " ".join(text.split())
    
    def remove_url(self, text:str)->str:
        pattern = r"((http|ftp|https):\/\/)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])?"
        return re.sub(pattern, "", text)
    
    def remove_cit(self, text:str)->str:
        # Remove citations like (Author, year) or (Author et al., year)
        text = re.sub(r"\s\([A-Z][a-z]+,\s[A-Z][a-z]?\.[^\)]*,\s\d{4}\)", "", text)
        # Remove bracketed numerical citations like [1], [2, 3], or [4-6]
        text = re.sub(r'\[\s*\d+(?:,\s*\d+)*(?:-\s*\d+)?\s*\]', '', text)
        # Remove citations like Author (Year)
        text = re.sub(r'\s[A-Z][a-zA-Z]*(?: et al\.)? \(\d{4}\)', '', text)
        return text
    
    def remove_html(self, text:str)->str:
        pattern = r"<[^>]+>"
        return re.sub(pattern, "", text)

    def lemma(self, doc)->str:
        lst = []
        lst = [token.lemma_.strip() for token in doc]
        return " ".join(lst)
    
    def pos(self, doc)->str:
        lst = []
        allowed_pos = ["PROPN", "NOUN", "VERB", "ADJ", "ADV"]
        lst = [token.text for token in doc if token.pos_ in allowed_pos]
                
        return " ".join(lst)


