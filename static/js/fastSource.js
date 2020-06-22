$(function() {
	ajax_post('/schemes', null, function(data, status){
		data.forEach(function (d) {
			$(".scheme-menu").append(
				$('<a/>')
				    .data(d)
					.addClass("w-inline-block w-scheme-link")
					.append(
						$('<img/>')
							.attr('title', d['SCHEME']['title'])
							.attr('src', d['SCHEME']['icon'])
							.addClass('db-icon')
					)
					.append(
						$('<div/>')
							.addClass('db-name w-scheme-text')
							.html(d['SCHEME']['title'])
							.append(
								$('<img />')
								.addClass('db-info-icon')
								.attr('scheme', d['scheme_name'])
								.attr('src', "static/img/info.svg")
							)
					)
			);
		});
		$(".scheme-menu .w-scheme-link")
		    .on('click', function(event) {
		    	var data = $(event.currentTarget).data();
		    	$("#jsGrid").trigger("update:fields", {'fields':data['LEVELS']});
		    	$(".scheme-menu .w-scheme-link").removeClass('w--current');
		    	$(event.currentTarget).addClass('w--current');
		    })
		    .first().click();
		$(".db-info-icon")
		    .on('click', function(event, d) {
		    	event.stopPropagation();
		    	var data = $(this).parent().parent().data();
		    	window.open(data.SCHEME.link, '_blank');
		    });
	});
	$(".raw-text").bind('input propertychange', function(event, d) {
		var query = $.trim($(".raw-text").val());
		if (query.length === 0) {
			$('.submit-raw')
			    .val('Please fill in raw text')
			    .prop('disabled', true);
		} else {
			var queries = query.split('\n');
			if (queries.length > 1000) {
			$('.submit-raw')
			    .val('Too many queries (>1000 lines)')
			    .prop('disabled', true);
			} else {
				$('.submit-raw')
					.val('Submit')
					.prop('disabled', false);
			}
		}
	});
    $('.submit-raw').on('click', function(event, d){
    	event.preventDefault();
    	var query = $.trim($(".raw-text").val());
    	var data = {
    		'scheme': $(".w--current").data()['NAME'],
    		'q': query,
    	};
    	ajax_post('batch_query', data, function(data, query) {
            var grid_data = [];
            var queries = query['q'].split('\n');
            this.category = data;
            var parent=this;
            queries.forEach(function(q) {
            	if (parent.category[q].length > 0) {
					parent.category[q].forEach(function(c) {
						grid_data.push(c);
					});
            	} else {
            		grid_data.push({
            			'Raw': q,
            		})
            	}
            	return d;
            });
            $("#jsGrid").trigger("update:data", {'data':grid_data});
    	});
    });


    var clients = [
        { "Name": "Otto Clay", "Age": 25, "Country": 1, "Address": "Ap #897-1459 Quam Avenue", "Married": false },
        { "Name": "Connor Johnston", "Age": 45, "Country": 2, "Address": "Ap #370-4647 Dis Av.", "Married": true },
        { "Name": "Lacey Hess", "Age": 29, "Country": 3, "Address": "Ap #365-8835 Integer St.", "Married": false },
        { "Name": "Timothy Henson", "Age": 56, "Country": 1, "Address": "911-5143 Luctus Ave", "Married": true },
        { "Name": "Ramona Benton", "Age": 32, "Country": 3, "Address": "Ap #614-689 Vehicula Street", "Married": false }
    ];

    $("#jsGrid").jsGrid({
        width: "100%",
        height: "400px",
        sorting: true,
        data: [{}], //clients,
        fields: [
            { name: "Raw", type: "text", width: 150, validate: "required" },
		]
	}).on('update:fields', function(event, data){
		var fields = ['Raw']
		    .concat(data['fields'])
		    .concat(['Pr'])
		    .map(function(fld) {
                return { name: fld, type: 'text', width: 150 };
		    });
		$(this).jsGrid({
			fields: fields,
			data: [{}],
		});
	}).on('update:data', function(event, data) {
		var data = data['data'];
        $(this).jsGrid({
        	data:data,
        });
	})
});


ajax_post = function(url, data, callback) {
	this.data = data;
	var self = this;
	var base_url = "http://137.205.123.127/fastsource/";
	$.ajax({
		url: base_url.concat(url), 
		method: 'POST', 
		data: data, 
		complete: function(resp, status) {
			return callback(JSON.parse(resp.responseText), self.data);
		},
		fail: function(resp, status) {
			console.log(url);
		}, 
		error: function(resp, status) {
			$('.submit-raw')
			    .val('Service unavailable. Please refresh after it is ready.')
			    .prop('disabled', true);
		}, 
	});
}
