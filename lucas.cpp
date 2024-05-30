#include <iostream>
using namespace std;
int main(){
int number; 
string name;
    cout << "Hello user. What is your name?" << endl;
    cin >> name;
    cout << "Hello" << name << ". Pick a number 1-10" << endl;
    cin >> number;
    cout << number << "Was the correct answer! Good job" << endl; 
    return 0;
}