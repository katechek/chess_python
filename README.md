# Шахматы
Программа, которая позволяет играть в шахматы. Взаимодействие с программой происходит через консоль. Игровое поле изображается в виде восьми текстовых строк и дополнительных строк с буквенным обозначением столбцов. Программа должна предоставлять пользователю возможность поочередно вводить ходы за белых и черных.

В случае ошибки в ответ на ввод пользователя выводится сообщение вида:
• Wrong input format. 
• The piece cannot make the specified move. 

Реализованы методы классов, позволяющие подсчитать количество белых (num_white_pieces()) и черных фигур (num_black_pieces()) и баланс белых и черных, измеряемый в пешках (balance())

# Формат ввода
Программа взаимодействует со стандартным потоком ввода-вывода и управляется командами в консоли. В консоль передаются команды: draw и exit, или координаты хода в формате xi-zj, например e2-f3. Пользователь может вводить команды на любом шаге.

# Пример
![image](https://github.com/user-attachments/assets/7f548e5f-3978-4f14-9dd8-4838bb5c3fbc)

