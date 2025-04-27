'use strict';
$(document).ready(function () {
    const searchBtn = document.getElementById("keySearch");

    searchBtn.onclick = function () {
        const key = document.getElementById("keyInput");
        if (key.value)
        {
            document.location.href = ('?k='+key.value);
        }
        else{
            document.location.href = ('?');
        }
    }
});
