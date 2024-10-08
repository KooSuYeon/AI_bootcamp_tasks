import random

# 1부터 10까지의 난수 생성
# random의 randrange는 앞은 열린, 뒤는 닫힌 구간이기 때문에 (1,11)로 범위 설정
number = random.randrange(1, 11)
print("1과 10 사이의 숫자를 하나 정했습니다.")
print("이 숫자는 무엇일까요?")

option = ""
while True:
    guess = int(input("예상 숫자: "))

    if guess < 1 or guess > 10:
        print("입력한 숫자는 1 ~ 10 사이의 숫자여야 합니다. 다시 입력하세요.")
        continue
    elif guess > number:
        print("너무 큽니다. 다시 입력하세요.")
        continue
    elif guess < number:
        print("너무 작습니다. 다시 입력하세요.")
        continue
    else:
        print("정답입니다!")

    option = input("종료를 원한다면 z를 입력하고 재시작을 원한다면 r을 입력해주세요 >>> ")
    if option == "z":
        break
    elif option == "r":
        number = random.randrange(1, 11)
        print("1과 10 사이의 숫자를 하나 정했습니다.")
        print("이 숫자는 무엇일까요?")