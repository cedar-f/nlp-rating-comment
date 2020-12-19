// function myFunction() {
// 	var element = document.getElementById('myDIV');
// 	element.classList.add('mystyle');
// }

// function myFunction() {
// 	var element = document.getElementById('myDIV');
// 	element.classList.remove('mystyle');
// }
// $('#back-bd').removeClass('animation-body');
$(function () {
	$('#switch').change(function () {
		let status = $(this).prop('checked');
		if (status) {
			$('.animation-heart').addClass('animation-heart');
			$('#back-bd').addClass('animation-body');
		} else {
			$('.animation-heart').removeClass('animation-heart');
			$('#back-bd').removeClass('animation-body');
		}
	});
});
