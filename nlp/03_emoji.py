import emoji

text = "shivanand devakate is the no 2 data scinetiest😊👌🙌👍👍"

# remove emojis from the text
clean_text = emoji.replace_emoji(text, replace="")
print(clean_text)
