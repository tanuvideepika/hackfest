import { getDatabase, ref, child, get }
 from "https://www.gstatic.com/firebasejs/9.19.1/firebase-databse.js";

const dbRef = ref(getDatabase());
get(child(dbRef, `users/${userId}`)).then((snapshot) => {
  if (snapshot.exists()) {
    console.log(snapshot.val());
  } else {
    console.log("No data available");
  }
}).catch((error) => {
  console.error(error);
});