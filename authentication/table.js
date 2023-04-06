import { initializeApp } from "https://www.gstatic.com/firebasejs/9.19.1/firebase-app.js";
const firebaseConfig = {
    apiKey: "AIzaSyDTcw62ynYIZUM4uBwp8B3mkGADhuUtbco",
    authDomain: "maal-fukte-hai.firebaseapp.com",
    projectId: "maal-fukte-hai",
    storageBucket: "maal-fukte-hai.appspot.com",
    messagingSenderId: "322921098043",
    appId: "1:322921098043:web:2a081a864912ec43eba51e"
  };

  import{
    getDatabase, ref, child, onValue, get
  } from "https://www.gstatic.com/firebasejs/9.19.1/firebase-database.js";

  // Initialize Firebase
const app = initializeApp(firebaseConfig);
const db= getDatabase();
var sno= 0;
var tbody = document.getElementById("tbody1");

function AddItemToTable(time, mood, text){
    let trow= document.createElement("tr");
    let td1= document.createElement("td");
    let td2= document.createElement("td");
    let td3= document.createElement("td");
    let td4= document.createElement("td");

    td1.innerHTML= ++sno;
    td2.innerHTML= time;
    td3.innerHTML= mood;
    td4.innerHTML= text;

    trow.appendChild(td1);
    trow.appendChild(td2);
    trow.appendChild(td3);
    trow.appendChild(td4);

    tbody.appendChild(trow);
    
}

function AllToTable(rant) {
    sno= 0;
    tbody.innerHTML= "";
    rant.forEach(element => {
        AddItemToTable(element.time, element.mood, element.text);

    });
}

function GetAllData (){
    const dbRef = ref(db);
    get(child(dbRef, "rant"))
    .then((snapshot) => {
        var rants= [];

        snapshot.forEach(childSnapshot =>{
            rants.push(childSnapshot.val())
        });
        AllToTable(rants);
    });
}

function GetDataRealTime(){

    const dbRef= ref(db, "rant");

    onValue(dbRef, (snapshot) => {
        var rants= [];

        snapshot.forEach(childSnapshot =>{
            rants.push(childSnapshot.val())
        });
        AllToTable(rants); 
    })
}
window.onload = GetDataRealTime();