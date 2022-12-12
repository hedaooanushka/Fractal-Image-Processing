'''
1st iteration:
Read original image
Convert into array of pixel values
Apply dct
Read encoding file
for every row in encoding file:
    Find corresponding Domain block
    Resize to 4x4
    Multiply with contrast(s) and add to brightness (o)
    Replace domain with range
apply inverse dct on transformed image 1
store transformed image 1

2nd iteration:
Read transformed image 1
Convert into array of pixel values
Read encoding file
for every row in encoding file:
    replace 4x4 range block with corresponding domain 4x4 domain block
Store transformed image 2

'''