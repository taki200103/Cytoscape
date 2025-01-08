package org.cytoscape.sample.internal;

import org.cytoscape.work.TaskFactory;
import org.cytoscape.work.TaskIterator;

public class LinkPredictionTaskFactory implements TaskFactory {

    @Override
    public TaskIterator createTaskIterator() {
        // Trả về TaskIterator cho LinkPredictionTask
        return new TaskIterator(new LinkPredictionTask());
    }

    @Override
    public boolean isReady() {
        // Kiểm tra xem TaskFactory có sẵn sàng hay không
        return true;
    }
}
