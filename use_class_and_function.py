gender_list = ["male", "female"]

class Person:
    def __init__(self, name, gender, age):
        self.name = name
        while gender not in gender_list:
            print("잘못된 성별을 입력하셨습니다. 'male' 또는 'female'을 입력하세요.")
            gender = input("성별: ")
        self.gender = gender
        self.age = age
    
    def display(self):
        print(f"이름: {self.name}, 성별: {self.gender}")
        print(f"나이: {self.age}")

    def greet(self):

        if self.age > 19:
            print(f"안녕하세요, {self.name}! 성인이시군요!")
        else:
            print(f"안녕하세요, {self.name}! 미성년자이시군요!")


age = int(input("나이: "))
name = input("이름: ")
gender = input("성별: ")

member = Person(name, gender, age)
member.display()