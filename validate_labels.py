import os  
label_path = "labels/train"
invalid_found = False  
for file in os.listdir(label_path):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(label_path, file)) as f:
        for line in f:
            values = line.strip().split()  
            if len(values) < 7:
                print("Invalid label:", file) 
                invalid_found = True          
if not invalid_found:
    print("\nValidation complete! All labels are valid")
else:
    print("\nValidation complete! Some labels are invalid")