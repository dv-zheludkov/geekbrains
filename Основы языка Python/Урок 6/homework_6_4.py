class Car:
    def __init__(self, speed, color, name, is_police):
        self.speed = speed
        self.color = color
        self.name = name
        self.is_police = is_police
    def show_speed(self):
        return self.speed
    def go(self):
        print("Car go")
    def stop(self):
        print("Car stop")
    def turn(self, direction):
        print(f"Car turn {direction}")

class TownCar(Car):
    def show_speed(self):
        if self.speed > 60:
            return "Превышение скорости"
        else:
            return self.speed

class WorkCar(Car):
    def show_speed(self):
        if self.speed > 40:
            print("Превышение скорости")
        return self.speed

class SportCar(Car):
    pass

class PoliceCar(Car):
    pass

town_car1 = TownCar(50, "white", "KIA", False)
sport_car1 = SportCar(150, "white", "VAZ", False)
work_car1 = WorkCar(50, "white", "Audi", False)
police_car1 = PoliceCar(100, "white", "Ford", True)

town_car1.go()
print(town_car1.show_speed())
town_car1.stop()

sport_car1.go()
sport_car1.turn("left")
sport_car1.stop()

work_car1.go()
work_car1.turn("right")
print(work_car1.show_speed())
work_car1.stop()

police_car1.go()
print(police_car1.show_speed())
police_car1.stop()
