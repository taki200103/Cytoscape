package org.cytoscape.sample.internal;

import org.cytoscape.work.TaskFactory;
import org.cytoscape.work.TaskIterator;

public class NodeEmbeddingTaskFactory implements TaskFactory {

    @Override
    public TaskIterator createTaskIterator() {
        return new TaskIterator(new NodeEmbeddingTask());
    }

    @Override
    public boolean isReady() {
        return true;
    }
}
