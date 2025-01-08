package org.cytoscape.sample.internal;

import org.cytoscape.work.TaskIterator;
import org.cytoscape.work.TaskFactory;

import javax.swing.*;
import java.io.File;

public class AddNodeTaskFactory implements TaskFactory {
    private File selectedCsvFile;

    @Override
    public TaskIterator createTaskIterator() {
        selectedCsvFile = selectCsvFile();
        if (selectedCsvFile != null) {
            return new TaskIterator(new AddNodeTask(selectedCsvFile));
        }
        return null; // Không tạo Task nếu file không được chọn
    }

    @Override
    public boolean isReady() {
        return true; // Luôn sẵn sàng
    }

    private File selectCsvFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select a CSV File");
        fileChooser.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("CSV Files", "csv"));
        
        int userSelection = fileChooser.showOpenDialog(null);
        if (userSelection == JFileChooser.APPROVE_OPTION) {
            return fileChooser.getSelectedFile();
        }
        return null; // Trả về null nếu không chọn file
    }
}
