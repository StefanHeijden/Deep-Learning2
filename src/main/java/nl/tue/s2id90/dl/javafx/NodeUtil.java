package nl.tue.s2id90.dl.javafx;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;

/**
 *
 * @author huub
 */
public class NodeUtil {
    public static TableView createInfoView(Map<String,Object> infoMap) {
        Set<Entry<String, Object>> entrySet = infoMap.entrySet();
        
        TableView<Entry<String,Object>> table = new TableView<>();
        table.setItems(FXCollections.observableArrayList(entrySet));
        
        TableColumn<Entry<String,Object>,String> col1 = new TableColumn<>("Key");
        col1.setCellValueFactory(p -> new SimpleStringProperty(p.getValue().getKey()));
        TableColumn<Entry<String,Object>,String> col2 = new TableColumn<>("Value");
        col2.setCellValueFactory(p -> new SimpleObjectProperty(p.getValue().getValue()));
        
        table.getColumns().setAll(col1,col2);
        
        return table;
    }
}
