text="hi THIS IS SHVNAND DEVAKATE  ....... ,sdfdsfmsf9834950-498498034I AM ETHONISTIC DATA SCIENTIEST"
import string
lowered=text.lower()
ed=""
for i in lowered:
    if i in string.punctuation  or 48<=ord(i)<=57:
        
        continue
    else:
        ed+=i

        
print(ed)

