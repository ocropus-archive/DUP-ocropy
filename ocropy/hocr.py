# various hOCR-related utilities

# FIXME make this much more general

# generate the headers and footers

header_template = """
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" 
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"> 
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<title>OCR Results</title>
<meta name="Description" content="OCRopus Output" />
<meta name="ocr-system" content="ocropus-0.4" />
<meta name="ocr-capabilities" content="ocr_line ocr_page" />
</head>
<body>
"""

footer_template = """
</body>
</html>
"""

def header():
    return header_template

def footer():
    return footer_template

