$( document ).ready(function() {
    $("#images-eval").on("submit", function(event) {   
        combine_selected_value("mix")
        combine_selected_value("rico_redraw")
        combine_selected_value("rawi")
    })

    $("#images-eval input[name='query']").val($("#search-box input[name='query']").val())
    $("#images-eval input[name='user']").val($("#search-box select[name='user']").val())
});

function combine_selected_value(name) {
    var ids_arr=[-1];
    var paths_arr=['path'];

    $("input:checked[name='" + name + "_selected_images[]']").each(function(){
        ids_arr.push($(this).attr("value-id"));
        paths_arr.push($(this).attr("value-path"));
    });

    $("#" + name + "_selected_image_ids").val(ids_arr.join(','));
    $("#" + name + "_selected_image_paths").val(paths_arr.join(','));
}
