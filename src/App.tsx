import { Grid, GridItem, Show } from '@chakra-ui/react';
import NavBar from './components/NavBar';
import ChapterGrid from './components/ChapterGrid';

function App() {
  return (
    <Grid
      templateAreas={{
        base: `'nav' 'main'`,
        lg: `'nav nav' 'aside main'`,
      }}>
      <GridItem area='nav'><NavBar/></GridItem>
      <Show above='lg'>
        <GridItem area='aside' bg='gold'>Aside</GridItem>
      </Show>
      <GridItem area='main' bg='dodgerblue'><ChapterGrid/></GridItem>
    </Grid>
  );
}

export default App;