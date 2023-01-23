# https://github.com/sangHa0411/CloneDetection/blob/main/utils/preprocessor.py
import re


class FunctionPreprocessor:
    def __init__(self):
        pass

    def get_function(self, code):
        results = []
        fn_list = re.findall("\ndef [a-zA-Z0-9_]+\(", code)

        for fn in fn_list:
            results.append(fn[4:-1].strip())
        return results

    def determine_function(self, code, function_name):
        num = len(re.findall("[^a-zA-Z]" + function_name + "[^a-zA-Z]", code))
        return False if num <= 1 else True

    def delete_function(self, code, name):
        start_id, _ = re.search("def " + name, code).span()
        ptr = start_id

        while ptr < len(code) - 1:
            if code[ptr] == "\n" and re.search("[a-zA-Z]", code[ptr + 1]) is not None:
                break
            ptr += 1

        if ptr != len(code) - 1:
            end_id = ptr
            code = code[:start_id] + code[end_id:]

        return code

    def preprocess(self, code):
        code = "\n" + code
        fn_list = self.get_function(code)
        if len(fn_list) == 0:
            return code

        for fn in fn_list:
            flag = self.determine_function(code, fn)

            if flag == False:
                code = self.delete_function(code, fn)

        return code

    def __call__(self, datasets):
        code1_list = []
        code2_list = []

        size = len(datasets["code1"])
        for i in range(size):
            code1 = self.preprocess(datasets["code1"][i])
            code2 = self.preprocess(datasets["code2"][i])

            code1_list.append(code1)
            code2_list.append(code2)

        datasets["code1"] = code1_list
        datasets["code2"] = code2_list
        return datasets


class AnnotationPreprocessor:
    def __init__(self):
        pass

    def search(self, sen_list, string):
        for i, sen in enumerate(sen_list):
            if string in sen:
                return i
        return -1

    def delete_annotation_block(self, code, string):
        sens = [sen for sen in code.split("\n")]

        start_id = self.search(sens, string)
        end_id = self.search(sens[start_id + 1 :], string)
        if end_id != -1:
            end_id += start_id + 1
            code = sens[:start_id] + sens[end_id + 1 :]
        else:
            code = sens[:start_id] + sens[start_id + 1 :]

        code = "\n".join(code)
        return code

    def delete_block(self, code, string):
        while string in code:
            code = self.delete_annotation_block(code, string)
        return code

    def delete_annotation(self, code):
        sens = code.split("\n")

        sens_processed = []
        for sen in sens:
            if "#" in sen:
                index = sen.index("#")
                sen = sen[:index]
            sens_processed.append(sen)

        return "\n".join(sens_processed)

    def delete_import(self, code):
        sens = code.split("\n")

        sens_processed = []
        for sen in sens:
            if "import" not in sen:
                sens_processed.append(sen)

        return "\n".join(sens_processed)

    def preprocess(self, code):
        code = self.delete_block(code, '"""')
        code = self.delete_block(code, "'''")
        code = self.delete_annotation(code)
        code = self.delete_import(code)
        code = re.sub("\s+", " ", code).strip()
        return code

    def __call__(self, datasets):
        code1_list = []
        code2_list = []

        size = len(datasets["code1"])
        for i in range(size):
            code1 = self.preprocess(datasets["code1"][i])
            code2 = self.preprocess(datasets["code2"][i])

            code1_list.append(code1)
            code2_list.append(code2)

        datasets["code1"] = code1_list
        datasets["code2"] = code2_list
        return datasets
